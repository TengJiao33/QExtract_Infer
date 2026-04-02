"""
Ring Buffer KV-Cache 功能测试

测试覆盖:
- 基本 append/get 操作
- Sink 区域保留验证
- Ring Buffer 环绕覆盖
- 满缓冲区行为
- 显存占用计算
"""

import torch
import pytest


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用, 跳过")
    return "cuda"


class TestRingBufferKVCache:
    """Ring Buffer KV-Cache 功能测试 (使用 PyTorch 回退实现)"""

    def _make_cache(self, **kwargs):
        from qextract.kv_cache import QExtractKVCache
        defaults = dict(
            num_layers=2,
            num_kv_heads=8,
            head_dim=128,
            sink_size=4,
            window_size=8,  # 小窗口方便测试
        )
        defaults.update(kwargs)
        return QExtractKVCache(**defaults)

    def test_basic_append_get(self, device):
        """基本的追加和读取"""
        cache = self._make_cache()

        for step in range(3):
            for layer in range(2):
                k = torch.randn(8, 128, dtype=torch.float16, device=device)
                v = torch.randn(8, 128, dtype=torch.float16, device=device)
                cache.append(k, v, layer)

        assert cache.valid_length == 3

        k_out, v_out = cache.get_kv(0)
        assert k_out.shape == (3, 8, 128)
        assert v_out.shape == (3, 8, 128)

    def test_sink_preservation(self, device):
        """验证 Sink 区域在 Ring Buffer 环绕后仍被保留"""
        cache = self._make_cache(sink_size=4, window_size=4)

        # 记录 sink token 的值
        sink_keys = []
        for step in range(4):  # 写入 4 个 sink token
            k = torch.full((8, 128), fill_value=float(step),
                           dtype=torch.float16, device=device)
            v = torch.zeros(8, 128, dtype=torch.float16, device=device)
            sink_keys.append(k.clone())
            for layer in range(2):
                cache.append(k, v, layer)

        # 再写入大量 token 触发 Ring Buffer 环绕
        for step in range(20):
            k = torch.full((8, 128), fill_value=100.0 + step,
                           dtype=torch.float16, device=device)
            v = torch.zeros(8, 128, dtype=torch.float16, device=device)
            for layer in range(2):
                cache.append(k, v, layer)

        # 验证 Sink 仍在
        assert cache.valid_length == 8  # 4 sink + 4 window

        k_out, _ = cache.get_kv(0)
        for i in range(4):
            torch.testing.assert_close(
                k_out[i], sink_keys[i],
                msg=f"Sink token {i} 被错误覆盖！"
            )

    def test_ring_buffer_wrap_around(self, device):
        """验证 Ring Buffer 正确环绕覆盖"""
        cache = self._make_cache(sink_size=2, window_size=4)

        # 写入 10 个 token (2 sink + 超出 window 的部分)
        for step in range(10):
            k = torch.full((8, 128), fill_value=float(step),
                           dtype=torch.float16, device=device)
            v = torch.zeros(8, 128, dtype=torch.float16, device=device)
            for layer in range(2):
                cache.append(k, v, layer)

        assert cache.valid_length == 6  # 2 sink + 4 window

        k_out, _ = cache.get_kv(0)

        # Sink (token 0, 1)
        assert k_out[0][0][0].item() == 0.0
        assert k_out[1][0][0].item() == 1.0

        # Ring Buffer 应包含最新的 4 个 token (6, 7, 8, 9)
        for i, expected_val in enumerate([6.0, 7.0, 8.0, 9.0]):
            assert k_out[2 + i][0][0].item() == expected_val, \
                f"Ring position {i}: 期望 {expected_val}, 实际 {k_out[2+i][0][0].item()}"

    def test_valid_length_progression(self, device):
        """验证 valid_length 的渐进增长"""
        cache = self._make_cache(sink_size=4, window_size=8)

        lengths = []
        for step in range(20):
            for layer in range(2):
                k = torch.randn(8, 128, dtype=torch.float16, device=device)
                v = torch.randn(8, 128, dtype=torch.float16, device=device)
                cache.append(k, v, layer)
            lengths.append(cache.valid_length)

        # 前 12 个 token: 长度应线性增长
        for i in range(12):
            assert lengths[i] == i + 1

        # 之后: 长度固定为 4 + 8 = 12
        for i in range(12, 20):
            assert lengths[i] == 12

    def test_reset(self, device):
        """验证重置功能"""
        cache = self._make_cache()

        for step in range(5):
            for layer in range(2):
                k = torch.randn(8, 128, dtype=torch.float16, device=device)
                v = torch.randn(8, 128, dtype=torch.float16, device=device)
                cache.append(k, v, layer)

        assert cache.valid_length == 5

        cache.reset()
        assert cache.valid_length == 0

    def test_memory_usage(self, device):
        """验证显存占用计算"""
        cache = self._make_cache(
            num_layers=36,    # Qwen3-4B
            num_kv_heads=8,
            head_dim=128,
            sink_size=4,
            window_size=4092,
        )

        expected_bytes = (
            36  # layers
            * (4 + 4092)  # total_size
            * 8  # kv_heads
            * 128  # head_dim
            * 2  # FP16 bytes
            * 2  # K + V
        )

        assert cache.memory_usage_bytes() == expected_bytes
        mem_mb = expected_bytes / (1024 * 1024)
        print(f"Qwen3-4B KV-Cache 预分配: {mem_mb:.1f} MB")

    def test_repr(self, device):
        """测试 repr 输出"""
        cache = self._make_cache()
        repr_str = repr(cache)
        assert "QExtractKVCache" in repr_str
        assert "sink=4" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
