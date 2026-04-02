/**
 * QExtract-Infer: 融合 SwiGLU 激活函数 CUDA 算子
 *
 * SwiGLU(gate, up) = SiLU(gate) * up = (gate * sigmoid(gate)) * up
 *
 * 原生 PyTorch 实现 (激活部分):
 *   act = F.silu(gate)       // 读 gate, 写 act (2 次显存访问)
 *   output = act * up        // 读 act, up, 写 output (3 次显存访问)
 *   总计: 5 次显存访问
 *
 * 融合实现:
 *   在一个 Kernel 中读 gate + up, 在寄存器内计算, 一次写回 output
 *   总计: 2 次读 + 1 次写 = 3 次显存访问
 *
 * 注意: 线性层的矩阵乘法由 cuBLAS 处理, 我们只融合激活函数部分
 */

#include "common.h"

// ─── 融合 SwiGLU Kernel ───
// output[i] = silu(gate[i]) * up[i]
//           = gate[i] * sigmoid(gate[i]) * up[i]
__global__ void fused_swiglu_kernel(
    const half_t* __restrict__ gate,      // [batch, intermediate_size]
    const half_t* __restrict__ up,        // [batch, intermediate_size]
    half_t* __restrict__ output,          // [batch, intermediate_size]
    const int n                            // total elements = batch * intermediate_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_g = g / (1.0f + expf(-g));

    output[idx] = __float2half(silu_g * u);
}

// ─── float4 向量化版本 ───
// 一次处理 8 个 half 值 (128 bit)
__global__ void fused_swiglu_vec_kernel(
    const float4* __restrict__ gate,
    const float4* __restrict__ up,
    float4* __restrict__ output,
    const int vec_count                    // total_elements / 8
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_count) return;

    float4 g_vec = gate[idx];
    float4 u_vec = up[idx];

    const half_t* g_h = reinterpret_cast<const half_t*>(&g_vec);
    const half_t* u_h = reinterpret_cast<const half_t*>(&u_vec);

    float4 o_vec;
    half_t* o_h = reinterpret_cast<half_t*>(&o_vec);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float g = __half2float(g_h[i]);
        float u = __half2float(u_h[i]);
        float silu_g = g / (1.0f + expf(-g));
        o_h[i] = __float2half(silu_g * u);
    }

    output[idx] = o_vec;
}

// ─── 融合 GeGLU Kernel (备选，某些模型变体使用) ───
__global__ void fused_geglu_kernel(
    const half_t* __restrict__ gate,
    const half_t* __restrict__ up,
    half_t* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float kAlpha = 0.7978845608f;  // sqrt(2/π)
    const float kBeta = 0.044715f;
    float gelu_g = 0.5f * g * (1.0f + tanhf(kAlpha * (g + kBeta * g * g * g)));

    output[idx] = __float2half(gelu_g * u);
}

// ─── C++ 接口函数 ───
torch::Tensor fused_swiglu(
    torch::Tensor gate,
    torch::Tensor up
) {
    CHECK_CUDA_INPUT(gate);
    CHECK_CUDA_INPUT(up);

    TORCH_CHECK(gate.dtype() == torch::kFloat16,
                "gate must be float16, got ", gate.dtype());
    TORCH_CHECK(gate.sizes() == up.sizes(),
                "gate and up must have same shape");

    auto output = torch::empty_like(gate);
    const int n = gate.numel();

    // 使用向量化版本 (如果元素数能被 8 整除)
    if (n % 8 == 0) {
        const int vec_count = n / 8;
        const int threads = 256;
        const int blocks = (vec_count + threads - 1) / threads;

        fused_swiglu_vec_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const float4*>(up.data_ptr<at::Half>()),
            reinterpret_cast<float4*>(output.data_ptr<at::Half>()),
            vec_count
        );
    } else {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;

        fused_swiglu_kernel<<<blocks, threads>>>(
            reinterpret_cast<const half_t*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half_t*>(up.data_ptr<at::Half>()),
            reinterpret_cast<half_t*>(output.data_ptr<at::Half>()),
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}
