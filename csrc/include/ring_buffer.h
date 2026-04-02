#pragma once

/**
 * QExtract-Infer: Ring Buffer KV-Cache class declaration
 * Implementation in kv_cache/ring_buffer.cu
 *
 * This header is designed to be includable from both .cpp and .cu files.
 * CUDA-specific types (__half etc.) are only used via opaque pointers.
 */

#include <torch/extension.h>
#include <vector>
#include <utility>

// CUDA runtime types needed for cudaMalloc/cudaFree etc.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ─── Macros (subset from common.h, safe for .cpp) ───
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));  \
        }                                                                 \
    } while (0)
#endif

#ifndef CHECK_CUDA_INPUT
#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)
#endif

constexpr int MAX_THREADS_PER_BLOCK_RB = 1024;

using half_t = __half;


class RingBufferKVCache {
public:
    RingBufferKVCache(
        int num_layers,
        int num_kv_heads,
        int head_dim,
        int sink_size,
        int window_size
    );

    ~RingBufferKVCache();

    void append(torch::Tensor new_k, torch::Tensor new_v, int layer_idx);
    std::pair<torch::Tensor, torch::Tensor> get_kv(int layer_idx);
    int get_valid_length() const;
    int get_total_written() const;
    void reset();

private:
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int sink_size_;
    int window_size_;
    int total_size_;
    int write_count_;

    std::vector<half_t*> k_caches_;
    std::vector<half_t*> v_caches_;
    int* d_indices_;
};
