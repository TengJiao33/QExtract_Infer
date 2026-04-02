"""
SwiGLU 融合算子数值对齐测试

对比 QExtract 融合实现与 PyTorch 原生实现
"""

import torch
import pytest


def swiglu_pytorch_ref(gate: torch.Tensor, up: torch.Tensor):
    """PyTorch 原生 SwiGLU 参考实现"""
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用, 跳过")
    return "cuda"


class TestFusedSwiGLU:
    """融合 SwiGLU 数值对齐测试"""

    def _get_fused_fn(self):
        try:
            from qextract._C import fused_swiglu
            return fused_swiglu
        except ImportError:
            pytest.skip("CUDA 后端未编译, 跳过")

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_qwen3_4b(self, device, batch_size):
        """Qwen3-4B: intermediate_size=9728"""
        intermediate_size = 9728
        fused_swiglu = self._get_fused_fn()

        gate = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)
        up = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)

        ref = swiglu_pytorch_ref(gate, up)
        ours = fused_swiglu(gate, up)

        torch.testing.assert_close(ours, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_qwen3_8b(self, device, batch_size):
        """Qwen3-8B: intermediate_size=12288"""
        intermediate_size = 12288
        fused_swiglu = self._get_fused_fn()

        gate = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)
        up = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)

        ref = swiglu_pytorch_ref(gate, up)
        ours = fused_swiglu(gate, up)

        torch.testing.assert_close(ours, ref, atol=1e-2, rtol=1e-2)

    def test_various_sizes(self, device):
        """测试多种 intermediate_size"""
        fused_swiglu = self._get_fused_fn()

        for size in [256, 512, 1024, 2048, 4096, 8192, 9728, 12288]:
            gate = torch.randn(2, size, dtype=torch.float16, device=device)
            up = torch.randn(2, size, dtype=torch.float16, device=device)

            ref = swiglu_pytorch_ref(gate, up)
            ours = fused_swiglu(gate, up)

            torch.testing.assert_close(
                ours, ref, atol=1e-2, rtol=1e-2,
                msg=f"size={size} 时数值不对齐"
            )

    def test_extreme_values(self, device):
        """测试极端输入值 (大值/小值/零)"""
        fused_swiglu = self._get_fused_fn()
        size = 1024

        # 大值
        gate = torch.randn(2, size, dtype=torch.float16, device=device) * 10
        up = torch.randn(2, size, dtype=torch.float16, device=device) * 10
        ref = swiglu_pytorch_ref(gate, up)
        ours = fused_swiglu(gate, up)
        # 大值下容差放宽
        torch.testing.assert_close(ours, ref, atol=5e-1, rtol=1e-1)

        # 接近零
        gate = torch.randn(2, size, dtype=torch.float16, device=device) * 0.01
        up = torch.randn(2, size, dtype=torch.float16, device=device) * 0.01
        ref = swiglu_pytorch_ref(gate, up)
        ours = fused_swiglu(gate, up)
        torch.testing.assert_close(ours, ref, atol=1e-3, rtol=1e-2)

    def test_1d_input(self, device):
        """测试 1D 输入"""
        fused_swiglu = self._get_fused_fn()

        gate = torch.randn(4096, dtype=torch.float16, device=device)
        up = torch.randn(4096, dtype=torch.float16, device=device)

        ref = swiglu_pytorch_ref(gate, up)
        ours = fused_swiglu(gate, up)

        torch.testing.assert_close(ours, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
