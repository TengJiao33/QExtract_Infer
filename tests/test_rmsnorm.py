"""
RMSNorm 融合算子数值对齐测试

对比 QExtract 融合实现与 PyTorch 原生实现的数值精度
确保算子替换不影响模型输出
"""

import torch
import pytest


def rmsnorm_pytorch_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """PyTorch 原生 RMSNorm 参考实现"""
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(input_dtype)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用, 跳过")
    return "cuda"


class TestFusedRMSNorm:
    """融合 RMSNorm 数值对齐测试"""

    def _get_fused_fn(self):
        try:
            from qextract._C import fused_rmsnorm
            return fused_rmsnorm
        except ImportError:
            pytest.skip("CUDA 后端未编译, 跳过")

    # ── Qwen3-4B 配置: hidden_size=2560 ──
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_qwen3_4b(self, device, batch_size):
        """Qwen3-4B 配置下的数值对齐"""
        hidden_size = 2560
        fused_rmsnorm = self._get_fused_fn()

        x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
        weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

        ref_output = rmsnorm_pytorch_ref(x, weight)
        our_output = fused_rmsnorm(x, weight, 1e-6)

        torch.testing.assert_close(our_output, ref_output, atol=1e-2, rtol=1e-2)

    # ── Qwen3-8B 配置: hidden_size=4096 ──
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_qwen3_8b(self, device, batch_size):
        """Qwen3-8B 配置下的数值对齐"""
        hidden_size = 4096
        fused_rmsnorm = self._get_fused_fn()

        x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
        weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

        ref_output = rmsnorm_pytorch_ref(x, weight)
        our_output = fused_rmsnorm(x, weight, 1e-6)

        torch.testing.assert_close(our_output, ref_output, atol=1e-2, rtol=1e-2)

    def test_various_hidden_sizes(self, device):
        """测试多种 hidden_size 以覆盖向量化和非向量化路径"""
        fused_rmsnorm = self._get_fused_fn()

        for hidden_size in [128, 256, 512, 768, 1024, 2048, 2560, 4096]:
            x = torch.randn(2, hidden_size, dtype=torch.float16, device=device)
            weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

            ref = rmsnorm_pytorch_ref(x, weight)
            ours = fused_rmsnorm(x, weight, 1e-6)

            torch.testing.assert_close(
                ours, ref, atol=1e-2, rtol=1e-2,
                msg=f"hidden_size={hidden_size} 时数值不对齐"
            )

    def test_eps_sensitivity(self, device):
        """测试不同 eps 值"""
        fused_rmsnorm = self._get_fused_fn()
        hidden_size = 2560

        x = torch.randn(4, hidden_size, dtype=torch.float16, device=device)
        weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

        for eps in [1e-5, 1e-6, 1e-8]:
            ref = rmsnorm_pytorch_ref(x, weight, eps)
            ours = fused_rmsnorm(x, weight, eps)
            torch.testing.assert_close(ours, ref, atol=1e-2, rtol=1e-2)

    def test_3d_input(self, device):
        """测试 3D 输入 [batch, seq_len, hidden_size]"""
        fused_rmsnorm = self._get_fused_fn()
        batch, seq_len, hidden_size = 2, 128, 2560

        x = torch.randn(batch, seq_len, hidden_size, dtype=torch.float16, device=device)
        weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

        # 展平 → 计算 → 还原
        x_flat = x.reshape(-1, hidden_size)
        ref = rmsnorm_pytorch_ref(x_flat, weight).reshape_as(x)
        ours = fused_rmsnorm(x_flat, weight, 1e-6).reshape_as(x)

        torch.testing.assert_close(ours, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
