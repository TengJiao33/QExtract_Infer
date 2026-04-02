"""
QExtract-Infer: Ring Buffer KV-Cache Python 封装层
兼容 HuggingFace transformers 的 Cache 接口
"""

import torch
from typing import Optional, Tuple, List


class QExtractKVCache:
    """
    StreamingLLM 风格的静态连续环形 KV-Cache

    特点:
    - O(1) 显存复杂度: 永不超出预分配的缓冲区
    - Attention Sink: 保留前 sink_size 个 Token 的 KV (注意力汇聚点)
    - Ring Buffer: 后续 Token 循环覆盖最旧的 Token

    用法:
        cache = QExtractKVCache(
            num_layers=36,     # Qwen3-4B
            num_kv_heads=8,    # GQA
            head_dim=128,
            sink_size=4,
            window_size=4092,  # 总容量 = 4 + 4092 = 4096
        )
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        sink_size: int = 4,
        window_size: int = 4092,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.window_size = window_size
        self.total_size = sink_size + window_size
        self.device = device
        self.dtype = dtype

        self.write_count = 0

        # 尝试使用 C++ 后端
        self._cpp_backend = None
        try:
            from qextract._C import RingBufferKVCache as CppCache
            self._cpp_backend = CppCache(
                num_layers, num_kv_heads, head_dim, sink_size, window_size
            )
            print("[QExtract KVCache] 使用 C++ CUDA 后端")
        except ImportError:
            # 回退到纯 PyTorch 实现
            print("[QExtract KVCache] C++ 后端不可用, 使用 PyTorch 回退实现")
            self._init_pytorch_fallback()

    def _init_pytorch_fallback(self):
        """纯 PyTorch 实现 (性能较低但功能等价, 用于开发和测试)"""
        shape = (self.total_size, self.num_kv_heads, self.head_dim)
        self.k_caches = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.v_caches = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    def append(
        self,
        new_k: torch.Tensor,  # [num_kv_heads, head_dim]
        new_v: torch.Tensor,  # [num_kv_heads, head_dim]
        layer_idx: int,
    ):
        """追加新 Token 的 KV 到指定层"""
        if self._cpp_backend is not None:
            self._cpp_backend.append(new_k, new_v, layer_idx)
            if layer_idx == self.num_layers - 1:
                self.write_count += 1
            return

        # PyTorch 回退
        write_pos = self._get_write_pos()
        self.k_caches[layer_idx][write_pos] = new_k
        self.v_caches[layer_idx][write_pos] = new_v

        if layer_idx == self.num_layers - 1:
            self.write_count += 1

    def get_kv(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定层的有效 KV 序列

        返回:
            (k, v), 每个 shape = [valid_len, num_kv_heads, head_dim]
        """
        if self._cpp_backend is not None:
            return self._cpp_backend.get_kv(layer_idx)

        # PyTorch 回退
        indices = self._build_indices()
        k = self.k_caches[layer_idx][indices]
        v = self.v_caches[layer_idx][indices]
        return k, v

    def _get_write_pos(self) -> int:
        """计算当前写入位置"""
        if self.write_count < self.sink_size:
            return self.write_count
        ring_idx = (self.write_count - self.sink_size) % self.window_size
        return self.sink_size + ring_idx

    def _build_indices(self) -> torch.Tensor:
        """构建有效 Token 的索引序列"""
        indices = []

        # Sink 部分
        actual_sink = min(self.write_count, self.sink_size)
        indices.extend(range(actual_sink))

        # Ring Buffer 部分
        if self.write_count > self.sink_size:
            ring_filled = min(self.write_count - self.sink_size, self.window_size)

            if self.write_count - self.sink_size <= self.window_size:
                ring_start = 0
            else:
                ring_start = (self.write_count - self.sink_size) % self.window_size

            for i in range(ring_filled):
                ring_pos = (ring_start + i) % self.window_size
                indices.append(self.sink_size + ring_pos)

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    @property
    def valid_length(self) -> int:
        """当前缓存中有效 Token 数量"""
        if self._cpp_backend is not None:
            return self._cpp_backend.get_valid_length()
        if self.write_count <= self.sink_size:
            return self.write_count
        ring_filled = min(self.write_count - self.sink_size, self.window_size)
        return self.sink_size + ring_filled

    @property
    def max_capacity(self) -> int:
        """最大容量"""
        return self.total_size

    def memory_usage_bytes(self) -> int:
        """当前显存占用 (字节)"""
        per_element = 2  # FP16 = 2 bytes
        per_layer = self.total_size * self.num_kv_heads * self.head_dim * per_element * 2  # K+V
        return self.num_layers * per_layer

    def reset(self):
        """重置缓存"""
        self.write_count = 0
        if self._cpp_backend is not None:
            self._cpp_backend.reset()
        else:
            for l in range(self.num_layers):
                self.k_caches[l].zero_()
                self.v_caches[l].zero_()

    def __repr__(self):
        mem_mb = self.memory_usage_bytes() / (1024 * 1024)
        return (
            f"QExtractKVCache("
            f"layers={self.num_layers}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"sink={self.sink_size}, "
            f"window={self.window_size}, "
            f"valid={self.valid_length}/{self.total_size}, "
            f"mem={mem_mb:.1f}MB)"
        )
