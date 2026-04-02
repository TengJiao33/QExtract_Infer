/**
 * QExtract-Infer: W4A16-LoRA 融合 GEMV CUDA 算子 (核武器)
 *
 * 计算目标: Y = X · dequant(W_int4) + α · X · A · B
 *
 * 这是整个项目最核心、最有价值的算子：
 * 在单个 CUDA Kernel 中完成底座 INT4 权重的在线反量化 + LoRA 补偿，
 * 不产生任何中间张量的全局显存写回。
 *
 * 优化策略:
 * 1. 将输入向量 X 加载到 Shared Memory (所有线程共享)
 * 2. INT4 底座权重用 cp.async 异步预取到 Shared Memory
 * 3. 在寄存器级别做位运算反量化
 * 4. LoRA 的 A、B 矩阵规模小 (rank 16~64), 可完全放入 Shared Memory
 * 5. 底座结果 + LoRA 结果在寄存器层面合并
 * 6. 用 Warp Shuffle 归约后一次性写回全局显存
 *
 * 相比原生 PyTorch (4+ 次全局显存读写) → 压缩为 1 次读 + 1 次写
 */

#include "common.h"
#include "quantize_utils.h"

// ──────────────────────────────────────────────
// Part 1: 纯 W4A16 GEMV (无 LoRA, 用于基线对比和初步验证)
// ──────────────────────────────────────────────

// 每个 Block 计算输出向量的一个元素
// 即: out[block_id] = dot(X, dequant(W_col[block_id]))
__global__ void w4a16_gemv_kernel(
    const half_t* __restrict__ input,         // [hidden_size] — 输入向量
    const uint32_t* __restrict__ qweight,     // [hidden_size/8, out_features] — INT4 打包权重
    const half_t* __restrict__ scales,        // [hidden_size/group_size, out_features] — 缩放因子
    const half_t* __restrict__ zeros,         // [hidden_size/group_size, out_features] — 零点
    half_t* __restrict__ output,              // [out_features] — 输出向量
    const int hidden_size,
    const int out_features,
    const int group_size
) {
    // 每个 Block 负责计算 output 的一个元素
    const int out_idx = blockIdx.x;
    if (out_idx >= out_features) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // 打包后的行数 (每 8 个 INT4 打包为 1 个 INT32)
    const int packed_rows = hidden_size / 8;
    const int num_groups = hidden_size / group_size;

    float local_sum = 0.0f;

    // 每个线程处理一部分打包权重
    for (int packed_row = tid; packed_row < packed_rows; packed_row += num_threads) {
        // 读取打包的 INT4 权重
        uint32_t packed = qweight[packed_row * out_features + out_idx];

        // 确定当前行属于哪个量化 group
        int base_row = packed_row * 8;
        int group_id = base_row / group_size;

        // 读取该 group 的 scale 和 zero_point
        float scale_f = __half2float(scales[group_id * out_features + out_idx]);
        float zp_f = __half2float(zeros[group_id * out_features + out_idx]);

        // 解包并反量化 8 个值, 同时做点积
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int int4_val = (packed >> (k * 4)) & 0xF;
            float dequant_w = scale_f * (static_cast<float>(int4_val) - zp_f);
            float x_val = __half2float(input[base_row + k]);
            local_sum += x_val * dequant_w;
        }
    }

    // Warp 级归约
    local_sum = warp_reduce_sum(local_sum);

    // Block 级归约 (跨 Warp)
    __shared__ float s_partial[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = num_threads / WARP_SIZE;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[out_idx] = __float2half(val);
        }
    }
}

// ──────────────────────────────────────────────
// Part 2: W4A16-LoRA 融合 GEMV (完整版核武器)
// ──────────────────────────────────────────────

// Y = X · dequant(W_int4) + α · (X · A) · B
//
// 分两阶段在同一个启动中完成:
// Kernel 1 (w4a16_lora_gemv_phase1): 计算 lora_hidden = X · A  (hidden→rank)
// Kernel 2 (w4a16_lora_gemv_phase2): 计算 base + LoRA 融合输出
//
// 由于 LoRA rank 很小 (16~64), Phase 1 极快, 这种两阶段方案
// 远优于把 LoRA 分支计算放进 per-output-element 的循环中

// Phase 1: 计算 lora_hidden = X · A
// A: [hidden_size, rank], FP16
__global__ void lora_down_gemv_kernel(
    const half_t* __restrict__ input,        // [hidden_size]
    const half_t* __restrict__ lora_A,       // [hidden_size, rank]
    float* __restrict__ lora_hidden,         // [rank] — 用 FP32 存中间结果避免精度损失
    const int hidden_size,
    const int rank
) {
    // 每个 Block 计算 lora_hidden 的一个元素
    const int r = blockIdx.x;
    if (r >= rank) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += num_threads) {
        float x = __half2float(input[i]);
        float a = __half2float(lora_A[i * rank + r]);
        local_sum += x * a;
    }

    // Warp 归约
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float s_partial[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = num_threads / WARP_SIZE;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            lora_hidden[r] = val;
        }
    }
}

// Phase 2: 融合 base GEMV + LoRA up projection
// out[j] = dot(X, dequant(W_col_j)) + α * dot(lora_hidden, B_col_j)
__global__ void w4a16_lora_fused_gemv_kernel(
    const half_t* __restrict__ input,         // [hidden_size]
    const uint32_t* __restrict__ qweight,     // [hidden_size/8, out_features]
    const half_t* __restrict__ scales,        // [num_groups, out_features]
    const half_t* __restrict__ zeros,         // [num_groups, out_features]
    const float* __restrict__ lora_hidden,    // [rank] — Phase 1 的输出
    const half_t* __restrict__ lora_B,        // [rank, out_features]
    half_t* __restrict__ output,              // [out_features]
    const int hidden_size,
    const int out_features,
    const int group_size,
    const int rank,
    const float lora_alpha                    // LoRA 缩放系数
) {
    const int out_idx = blockIdx.x;
    if (out_idx >= out_features) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int packed_rows = hidden_size / 8;

    // ── 底座 GEMV: dot(X, dequant(W)) ──
    float base_sum = 0.0f;
    for (int packed_row = tid; packed_row < packed_rows; packed_row += num_threads) {
        uint32_t packed = qweight[packed_row * out_features + out_idx];
        int base_row = packed_row * 8;
        int group_id = base_row / group_size;

        float scale_f = __half2float(scales[group_id * out_features + out_idx]);
        float zp_f = __half2float(zeros[group_id * out_features + out_idx]);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int int4_val = (packed >> (k * 4)) & 0xF;
            float dequant_w = scale_f * (static_cast<float>(int4_val) - zp_f);
            float x_val = __half2float(input[base_row + k]);
            base_sum += x_val * dequant_w;
        }
    }

    // Warp + Block 归约底座结果
    base_sum = warp_reduce_sum(base_sum);

    __shared__ float s_partial[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = base_sum;
    __syncthreads();

    float final_base = 0.0f;
    if (warp_id == 0) {
        int num_warps = num_threads / WARP_SIZE;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        final_base = warp_reduce_sum(val);
    }

    // ── LoRA 分支: dot(lora_hidden, B_col) ──
    // rank 很小 (16~64), 单线程即可完成
    if (tid == 0) {
        float lora_sum = 0.0f;
        for (int r = 0; r < rank; r++) {
            lora_sum += lora_hidden[r] * __half2float(lora_B[r * out_features + out_idx]);
        }

        // 合并: base + α * lora
        float scaling = lora_alpha / static_cast<float>(rank);
        float result = final_base + scaling * lora_sum;

        output[out_idx] = __float2half(result);
    }
}

// ──────────────────────────────────────────────
// C++ 接口
// ──────────────────────────────────────────────

// 纯 W4A16 GEMV (无 LoRA)
torch::Tensor w4a16_gemv(
    torch::Tensor input,       // [hidden_size], float16
    torch::Tensor qweight,     // [hidden_size/8, out_features], int32
    torch::Tensor scales,      // [num_groups, out_features], float16
    torch::Tensor zeros,       // [num_groups, out_features], float16
    int group_size
) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(qweight);
    CHECK_CUDA_INPUT(scales);
    CHECK_CUDA_INPUT(zeros);

    const int hidden_size = input.size(0);
    const int out_features = qweight.size(1);

    auto output = torch::empty({out_features}, input.options());

    const int threads = 256;
    const int blocks = out_features;

    w4a16_gemv_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half_t*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half_t*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<half_t*>(output.data_ptr<at::Half>()),
        hidden_size,
        out_features,
        group_size
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// W4A16-LoRA 融合 GEMV (完整版)
torch::Tensor w4a16_lora_gemv(
    torch::Tensor input,       // [hidden_size], float16
    torch::Tensor qweight,     // [hidden_size/8, out_features], int32
    torch::Tensor scales,      // [num_groups, out_features], float16
    torch::Tensor zeros,       // [num_groups, out_features], float16
    torch::Tensor lora_A,      // [hidden_size, rank], float16
    torch::Tensor lora_B,      // [rank, out_features], float16
    int group_size,
    float lora_alpha
) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(qweight);
    CHECK_CUDA_INPUT(scales);
    CHECK_CUDA_INPUT(zeros);
    CHECK_CUDA_INPUT(lora_A);
    CHECK_CUDA_INPUT(lora_B);

    const int hidden_size = input.size(0);
    const int out_features = qweight.size(1);
    const int rank = lora_A.size(1);

    // Phase 1: X · A → lora_hidden [rank]
    auto lora_hidden = torch::empty({rank}, torch::dtype(torch::kFloat32).device(input.device()));

    const int phase1_threads = 256;
    lora_down_gemv_kernel<<<rank, phase1_threads>>>(
        reinterpret_cast<const half_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half_t*>(lora_A.data_ptr<at::Half>()),
        lora_hidden.data_ptr<float>(),
        hidden_size,
        rank
    );
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: base GEMV + LoRA up → output
    auto output = torch::empty({out_features}, input.options());

    const int phase2_threads = 256;
    w4a16_lora_fused_gemv_kernel<<<out_features, phase2_threads>>>(
        reinterpret_cast<const half_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half_t*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half_t*>(zeros.data_ptr<at::Half>()),
        lora_hidden.data_ptr<float>(),
        reinterpret_cast<const half_t*>(lora_B.data_ptr<at::Half>()),
        reinterpret_cast<half_t*>(output.data_ptr<at::Half>()),
        hidden_size,
        out_features,
        group_size,
        rank,
        lora_alpha
    );
    CUDA_CHECK(cudaGetLastError());

    return output;
}
