/**
 * QExtract-Infer: StreamingLLM Ring Buffer KV-Cache
 *
 * Scene-specific: IE tasks = long Prefill + short Decode
 * In this scenario, PagedAttention from general frameworks is overhead
 *
 * Design:
 * +-----------------------------------------------+
 * | Attention Sink (fixed first sink_size tokens)  |
 * +-----------------------------------------------+
 * | Ring Buffer (circular overwrite, O(1) memory)  |
 * | write_ptr ->                                   |
 * +-----------------------------------------------+
 *
 * Total capacity = sink_size + window_size
 * When writes exceed window_size, write_ptr wraps to overwrite oldest token
 * Attention Sink portion is never overwritten (StreamingLLM theory)
 */

#include "common.h"
#include "ring_buffer.h"

// ─── KV Cache Append Kernel ───
// Write new token K/V to Ring Buffer at write_ptr position
__global__ void kv_cache_append_kernel(
    half_t* __restrict__ k_cache,         // [total_size, num_heads, head_dim]
    half_t* __restrict__ v_cache,         // [total_size, num_heads, head_dim]
    const half_t* __restrict__ new_k,     // [num_heads, head_dim]
    const half_t* __restrict__ new_v,     // [num_heads, head_dim]
    const int write_pos,                  // write position (with sink offset + ring wrap)
    const int num_heads,
    const int head_dim
) {
    // Each thread handles one element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_heads * head_dim;
    if (idx >= total) return;

    const int cache_offset = write_pos * total + idx;
    k_cache[cache_offset] = new_k[idx];
    v_cache[cache_offset] = new_v[idx];
}

// ─── KV Cache Gather Kernel ───
// Gather valid KV sequence from Ring Buffer for Attention computation
// Output order: [sink_tokens..., ring_valid_tokens...]
__global__ void kv_cache_gather_kernel(
    const half_t* __restrict__ cache,     // [total_size, num_heads, head_dim]
    half_t* __restrict__ gathered,        // [seq_len, num_heads, head_dim]
    const int* __restrict__ indices,      // [seq_len] — physical position indices of valid tokens
    const int seq_len,
    const int stride                      // = num_heads * head_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= seq_len) return;

    const int element_idx = threadIdx.x;
    if (element_idx >= stride) return;

    const int src_pos = indices[token_idx];
    gathered[token_idx * stride + element_idx] = cache[src_pos * stride + element_idx];
}


// ──────────────────────────────────────────────
// RingBufferKVCache method implementations
// ──────────────────────────────────────────────

RingBufferKVCache::RingBufferKVCache(
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int sink_size,
    int window_size
) : num_layers_(num_layers),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    sink_size_(sink_size),
    window_size_(window_size),
    total_size_(sink_size + window_size),
    write_count_(0)
{
    const size_t stride = num_kv_heads * head_dim;
    const size_t layer_bytes = total_size_ * stride * sizeof(half_t);

    // Allocate K and V memory for each layer
    k_caches_.resize(num_layers);
    v_caches_.resize(num_layers);

    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaMalloc(&k_caches_[l], layer_bytes));
        CUDA_CHECK(cudaMalloc(&v_caches_[l], layer_bytes));
        CUDA_CHECK(cudaMemset(k_caches_[l], 0, layer_bytes));
        CUDA_CHECK(cudaMemset(v_caches_[l], 0, layer_bytes));
    }

    // Allocate index buffer (Device)
    CUDA_CHECK(cudaMalloc(&d_indices_, total_size_ * sizeof(int)));
}

RingBufferKVCache::~RingBufferKVCache() {
    for (int l = 0; l < num_layers_; l++) {
        cudaFree(k_caches_[l]);
        cudaFree(v_caches_[l]);
    }
    cudaFree(d_indices_);
}

void RingBufferKVCache::append(torch::Tensor new_k, torch::Tensor new_v, int layer_idx) {
    CHECK_CUDA_INPUT(new_k);
    CHECK_CUDA_INPUT(new_v);

    const int stride = num_kv_heads_ * head_dim_;
    int write_pos;

    if (write_count_ < sink_size_) {
        // Still filling Sink region
        write_pos = write_count_;
    } else {
        // Ring Buffer region
        int ring_idx = (write_count_ - sink_size_) % window_size_;
        write_pos = sink_size_ + ring_idx;
    }

    const int threads = 256;
    const int blocks = (stride + threads - 1) / threads;

    kv_cache_append_kernel<<<blocks, threads>>>(
        k_caches_[layer_idx],
        v_caches_[layer_idx],
        reinterpret_cast<const half_t*>(new_k.data_ptr<at::Half>()),
        reinterpret_cast<const half_t*>(new_v.data_ptr<at::Half>()),
        write_pos,
        num_kv_heads_,
        head_dim_
    );

    if (layer_idx == num_layers_ - 1) {
        write_count_++;
    }
}

std::pair<torch::Tensor, torch::Tensor> RingBufferKVCache::get_kv(int layer_idx) {
    const int stride = num_kv_heads_ * head_dim_;
    const int valid_len = get_valid_length();

    // Build index array
    std::vector<int> h_indices(valid_len);

    // Sink portion
    int actual_sink = std::min(write_count_, sink_size_);
    for (int i = 0; i < actual_sink; i++) {
        h_indices[i] = i;
    }

    // Ring Buffer portion
    if (write_count_ > sink_size_) {
        int ring_filled = std::min(write_count_ - sink_size_, window_size_);
        int ring_start;

        if (write_count_ - sink_size_ <= window_size_) {
            // Buffer not full, read sequentially from sink_size_
            ring_start = 0;
        } else {
            // Buffer full, read circularly from write_ptr
            ring_start = (write_count_ - sink_size_) % window_size_;
        }

        for (int i = 0; i < ring_filled; i++) {
            int ring_pos = (ring_start + i) % window_size_;
            h_indices[actual_sink + i] = sink_size_ + ring_pos;
        }
    }

    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(d_indices_, h_indices.data(),
                          valid_len * sizeof(int), cudaMemcpyHostToDevice));

    // Gather
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(torch::kCUDA);
    auto k_gathered = torch::empty({valid_len, num_kv_heads_, head_dim_}, opts);
    auto v_gathered = torch::empty({valid_len, num_kv_heads_, head_dim_}, opts);

    if (valid_len > 0) {
        const int threads = std::min(stride, MAX_THREADS_PER_BLOCK);
        kv_cache_gather_kernel<<<valid_len, threads>>>(
            k_caches_[layer_idx],
            reinterpret_cast<half_t*>(k_gathered.data_ptr<at::Half>()),
            d_indices_, valid_len, stride
        );
        kv_cache_gather_kernel<<<valid_len, threads>>>(
            v_caches_[layer_idx],
            reinterpret_cast<half_t*>(v_gathered.data_ptr<at::Half>()),
            d_indices_, valid_len, stride
        );
    }

    return {k_gathered, v_gathered};
}

int RingBufferKVCache::get_valid_length() const {
    if (write_count_ <= sink_size_) {
        return write_count_;
    }
    int ring_filled = std::min(write_count_ - sink_size_, window_size_);
    return sink_size_ + ring_filled;
}

int RingBufferKVCache::get_total_written() const { return write_count_; }

void RingBufferKVCache::reset() { write_count_ = 0; }
