/**
 * QExtract-Infer: StreamingLLM 风格的 Ring Buffer KV-Cache
 *
 * 场景特化: 信息抽取任务 (IE) = 长 Prefill + 短 Decode
 * 在这种场景下, 通用框架的 PagedAttention 动态分页是累赘
 *
 * 设计:
 * ┌───────────────────────────────────────────────┐
 * │ Attention Sink (固定前 sink_size 个 Token)     │
 * ├───────────────────────────────────────────────┤
 * │ Ring Buffer (循环覆盖, O(1) 显存复杂度)        │
 * │ write_ptr →                                   │
 * └───────────────────────────────────────────────┘
 *
 * 总容量 = sink_size + window_size
 * 当写入超过 window_size 时, write_ptr 绕回覆盖最旧的 Token
 * Attention Sink 部分永远不被覆盖 (StreamingLLM 理论)
 */

#include "common.h"

// ─── KV Cache Append Kernel ───
// 将新 Token 的 K/V 写入 Ring Buffer 的 write_ptr 位置
__global__ void kv_cache_append_kernel(
    half_t* __restrict__ k_cache,         // [total_size, num_heads, head_dim]
    half_t* __restrict__ v_cache,         // [total_size, num_heads, head_dim]
    const half_t* __restrict__ new_k,     // [num_heads, head_dim]
    const half_t* __restrict__ new_v,     // [num_heads, head_dim]
    const int write_pos,                  // 写入位置 (已考虑 sink offset + ring wrap)
    const int num_heads,
    const int head_dim
) {
    // 每个线程处理一个元素
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_heads * head_dim;
    if (idx >= total) return;

    const int cache_offset = write_pos * total + idx;
    k_cache[cache_offset] = new_k[idx];
    v_cache[cache_offset] = new_v[idx];
}

// ─── KV Cache Gather Kernel ───
// 从 Ring Buffer 中收集有效的 KV 序列用于 Attention 计算
// 输出顺序: [sink_tokens..., ring_valid_tokens...]
__global__ void kv_cache_gather_kernel(
    const half_t* __restrict__ cache,     // [total_size, num_heads, head_dim]
    half_t* __restrict__ gathered,        // [seq_len, num_heads, head_dim]
    const int* __restrict__ indices,      // [seq_len] — 有效 Token 的物理位置索引
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
// Ring Buffer 管理器 (C++ 类, 通过 PyBind11 暴露)
// ──────────────────────────────────────────────

class RingBufferKVCache {
public:
    RingBufferKVCache(
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

        // 为每一层分配 K 和 V 的显存
        k_caches_.resize(num_layers);
        v_caches_.resize(num_layers);

        for (int l = 0; l < num_layers; l++) {
            CUDA_CHECK(cudaMalloc(&k_caches_[l], layer_bytes));
            CUDA_CHECK(cudaMalloc(&v_caches_[l], layer_bytes));
            CUDA_CHECK(cudaMemset(k_caches_[l], 0, layer_bytes));
            CUDA_CHECK(cudaMemset(v_caches_[l], 0, layer_bytes));
        }

        // 分配索引缓冲区 (Host + Device)
        CUDA_CHECK(cudaMalloc(&d_indices_, total_size_ * sizeof(int)));
    }

    ~RingBufferKVCache() {
        for (int l = 0; l < num_layers_; l++) {
            cudaFree(k_caches_[l]);
            cudaFree(v_caches_[l]);
        }
        cudaFree(d_indices_);
    }

    // 追加一个新 Token 的 KV 到指定层
    void append(torch::Tensor new_k, torch::Tensor new_v, int layer_idx) {
        CHECK_CUDA_INPUT(new_k);
        CHECK_CUDA_INPUT(new_v);

        const int stride = num_kv_heads_ * head_dim_;
        int write_pos;

        if (write_count_ < sink_size_) {
            // 还在填充 Sink 区域
            write_pos = write_count_;
        } else {
            // Ring Buffer 区域
            int ring_idx = (write_count_ - sink_size_) % window_size_;
            write_pos = sink_size_ + ring_idx;
        }

        // 仅在 layer 0 时递增计数器, 避免多层重复计数
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

    // 获取指定层的有效 KV 序列
    // 返回: (k_gathered, v_gathered), shape = [valid_len, num_kv_heads, head_dim]
    std::pair<torch::Tensor, torch::Tensor> get_kv(int layer_idx) {
        const int stride = num_kv_heads_ * head_dim_;
        const int valid_len = get_valid_length();

        // 构建索引数组
        std::vector<int> h_indices(valid_len);

        // Sink 部分
        int actual_sink = std::min(write_count_, sink_size_);
        for (int i = 0; i < actual_sink; i++) {
            h_indices[i] = i;
        }

        // Ring Buffer 部分
        if (write_count_ > sink_size_) {
            int ring_filled = std::min(write_count_ - sink_size_, window_size_);
            int ring_start;

            if (write_count_ - sink_size_ <= window_size_) {
                // Buffer 未满, 从 sink_size_ 开始顺序读
                ring_start = 0;
            } else {
                // Buffer 已满, 从 write_ptr 开始环形读
                ring_start = (write_count_ - sink_size_) % window_size_;
            }

            for (int i = 0; i < ring_filled; i++) {
                int ring_pos = (ring_start + i) % window_size_;
                h_indices[actual_sink + i] = sink_size_ + ring_pos;
            }
        }

        // 传到 GPU
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

    int get_valid_length() const {
        if (write_count_ <= sink_size_) {
            return write_count_;
        }
        int ring_filled = std::min(write_count_ - sink_size_, window_size_);
        return sink_size_ + ring_filled;
    }

    int get_total_written() const { return write_count_; }

    void reset() { write_count_ = 0; }

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
