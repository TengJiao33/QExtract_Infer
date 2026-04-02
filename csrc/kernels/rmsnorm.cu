/**
 * QExtract-Infer: 融合 RMSNorm CUDA 算子
 *
 * 原生 PyTorch 实现需要 3 次全局显存读写：
 *   1. 读 x → 计算 x² → 写 variance
 *   2. 读 x, variance → 计算 rsqrt → 写 normalized
 *   3. 读 normalized, weight → 乘 → 写 output
 *
 * 本融合实现仅需 1 次读 + 1 次写：
 *   - 使用 float4 向量化加载 (128-bit)
 *   - Warp Shuffle 归约求方差
 *   - 寄存器内完成 rsqrt × weight
 *   - 一次性写回
 */

#include "common.h"

// ─── 融合 RMSNorm Kernel ───
// 每个 Block 处理输入矩阵的一行 (hidden_size 个元素)
// blockDim.x 应为 WARP_SIZE 的倍数
template <int HIDDEN_SIZE_PADDED>
__global__ void fused_rmsnorm_kernel(
    const half_t* __restrict__ input,     // [batch, hidden_size]
    const half_t* __restrict__ weight,    // [hidden_size]
    half_t* __restrict__ output,          // [batch, hidden_size]
    const int hidden_size,
    const float eps
) {
    // 每个 block 处理一行
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const half_t* row_input = input + row * hidden_size;
    half_t* row_output = output + row * hidden_size;

    // ── Step 1: 向量化加载 + 累加 x² ──
    // 每个线程处理 hidden_size / num_threads 个元素
    float local_sum_sq = 0.0f;

    for (int i = tid; i < hidden_size; i += num_threads) {
        float val = __half2float(row_input[i]);
        local_sum_sq += val * val;
    }

    // ── Step 2: Warp Shuffle 归约求 sum(x²) ──
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Block 内跨 Warp 归约 (通过 Shared Memory)
    __shared__ float s_partial[32];  // 最多 32 个 Warp
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_partial[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // 第一个 Warp 做最终归约
    float total_sum_sq = 0.0f;
    if (warp_id == 0) {
        int num_warps = num_threads / WARP_SIZE;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        total_sum_sq = warp_reduce_sum(val);
    }

    // 广播 rsqrt 结果到所有线程
    __shared__ float s_rsqrt;
    if (tid == 0) {
        float variance = total_sum_sq / static_cast<float>(hidden_size);
        s_rsqrt = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rms_scale = s_rsqrt;

    // ── Step 3: 归一化 × weight → 写回 ──
    for (int i = tid; i < hidden_size; i += num_threads) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        float normalized = val * rms_scale * w;
        row_output[i] = __float2half(normalized);
    }
}

// ─── float4 向量化版本 (适用于 hidden_size 能被 8 整除的情况) ───
__global__ void fused_rmsnorm_vec_kernel(
    const float4* __restrict__ input,     // reinterpret as float4
    const float4* __restrict__ weight,
    float4* __restrict__ output,
    const int hidden_size,                // 原始 hidden_size (half 计数)
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // float4 = 128 bit = 8 个 half
    const int vec_size = hidden_size / 8;

    const float4* row_in = input + row * vec_size;
    float4* row_out = output + row * vec_size;

    // ── Step 1: 向量化加载 + 累加 ──
    float local_sum_sq = 0.0f;

    for (int i = tid; i < vec_size; i += num_threads) {
        float4 v = row_in[i];
        // 将 float4 重新解释为 8 个 half
        const half_t* hp = reinterpret_cast<const half_t*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float f = __half2float(hp[j]);
            local_sum_sq += f * f;
        }
    }

    // ── Step 2: 归约 ──
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    __shared__ float s_partial[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = local_sum_sq;
    __syncthreads();

    __shared__ float s_rsqrt;
    if (warp_id == 0) {
        int num_warps = num_threads / WARP_SIZE;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        float total = warp_reduce_sum(val);
        if (lane_id == 0) {
            s_rsqrt = rsqrtf(total / static_cast<float>(hidden_size) + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_rsqrt;

    // ── Step 3: 归一化 × weight → 写回 ──
    for (int i = tid; i < vec_size; i += num_threads) {
        float4 v_in = row_in[i];
        float4 v_w = weight[i];
        const half_t* hp_in = reinterpret_cast<const half_t*>(&v_in);
        const half_t* hp_w = reinterpret_cast<const half_t*>(&v_w);

        float4 v_out;
        half_t* hp_out = reinterpret_cast<half_t*>(&v_out);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __half2float(hp_in[j]) * rms_scale * __half2float(hp_w[j]);
            hp_out[j] = __float2half(val);
        }

        row_out[i] = v_out;
    }
}

// ─── C++ 接口函数 ───
torch::Tensor fused_rmsnorm(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(weight);

    TORCH_CHECK(input.dtype() == torch::kFloat16,
                "input must be float16, got ", input.dtype());
    TORCH_CHECK(weight.dtype() == torch::kFloat16,
                "weight must be float16, got ", weight.dtype());

    // 支持任意前置维度: [..., hidden_size]
    const int hidden_size = input.size(-1);
    const int num_rows = input.numel() / hidden_size;

    TORCH_CHECK(weight.numel() == hidden_size,
                "weight size mismatch: ", weight.numel(), " vs ", hidden_size);

    auto output = torch::empty_like(input);

    // 选择 kernel 配置
    const int threads = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    // 确保线程数是 WARP_SIZE 的倍数
    const int threads_aligned = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // 如果 hidden_size 能被 8 整除，使用向量化版本
    if (hidden_size % 8 == 0 && hidden_size >= 256) {
        fused_rmsnorm_vec_kernel<<<num_rows, threads_aligned>>>(
            reinterpret_cast<const float4*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const float4*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<float4*>(output.data_ptr<at::Half>()),
            hidden_size,
            eps
        );
    } else {
        fused_rmsnorm_kernel<0><<<num_rows, threads_aligned>>>(
            reinterpret_cast<const half_t*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half_t*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half_t*>(output.data_ptr<at::Half>()),
            hidden_size,
            eps
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}
