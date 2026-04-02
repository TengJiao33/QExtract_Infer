#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ─── CUDA 错误检查宏 ───
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));   \
        }                                                                 \
    } while (0)

// ─── 输入张量校验宏 ───
#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA_INPUT(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)

// ─── 常用常量 ───
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// ─── 类型别名 ───
using half_t = __half;
using half2_t = __half2;

// ─── Warp 级归约工具 ───
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ─── 向量化加载辅助 ───
// float4 = 128 bit = 4 x float = 8 x half
struct Half8 {
    half2_t data[4];  // 4 x half2 = 8 x half = 128 bit
};
