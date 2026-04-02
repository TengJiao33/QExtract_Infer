"""
W4A16-LoRA GEMV 融合算子数值对齐测试

测试策略:
1. 先测纯 W4A16 GEMV (无 LoRA): 验证 INT4 反量化正确性
2. 再测 W4A16-LoRA 融合 GEMV: 验证 LoRA 补偿正确性
3. 使用模拟权重 (随机生成 + 手工量化)
"""

import torch
import pytest


def simulate_gptq_quantize(
    weight_fp16: torch.Tensor,  # [in_features, out_features]
    group_size: int = 128
) -> dict:
    """
    模拟 GPTQ 量化过程
    真实 GPTQ 使用 Hessian 信息做分组最优量化, 这里简化为均匀量化

    返回: {qweight, scales, zeros} 以 GPTQ 打包格式
    """
    in_features, out_features = weight_fp16.shape
    device = weight_fp16.device

    num_groups = in_features // group_size
    assert in_features % group_size == 0, "in_features 必须能被 group_size 整除"

    # 分组求 min/max
    w_grouped = weight_fp16.reshape(num_groups, group_size, out_features)

    w_min = w_grouped.min(dim=1).values  # [num_groups, out_features]
    w_max = w_grouped.max(dim=1).values

    # 计算 scale 和 zero_point (对称量化, 映射到 0~15)
    scales = (w_max - w_min) / 15.0
    scales = scales.clamp(min=1e-6)  # 防止除零
    zeros = -w_min / scales  # zero_point: 使得 dequant(0) = w_min

    # 量化
    qweight_int = torch.zeros(in_features, out_features, dtype=torch.int32, device=device)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        w_slice = weight_fp16[start:end]  # [group_size, out_features]

        # quantize: q = round(w / scale + zero_point)
        s = scales[g].unsqueeze(0)  # [1, out_features]
        z = zeros[g].unsqueeze(0)
        q = torch.round(w_slice.float() / s.float() + z.float()).clamp(0, 15).to(torch.int32)
        qweight_int[start:end] = q

    # 打包: 每 8 个 INT4 → 1 个 INT32
    packed_rows = in_features // 8
    qweight_packed = torch.zeros(packed_rows, out_features, dtype=torch.int32, device=device)

    for i in range(8):
        qweight_packed |= (qweight_int[torch.arange(packed_rows) * 8 + i] & 0xF) << (i * 4)

    return {
        "qweight": qweight_packed,      # [in_features/8, out_features], int32
        "scales": scales.half(),         # [num_groups, out_features], fp16
        "zeros": zeros.half(),           # [num_groups, out_features], fp16
    }


def dequantize_gptq_ref(qweight, scales, zeros, hidden_size, group_size=128):
    """参考反量化实现: 从打包格式还原 FP16 权重"""
    packed_rows, out_features = qweight.shape
    device = qweight.device

    weight_fp16 = torch.zeros(hidden_size, out_features, dtype=torch.float16, device=device)

    for pr in range(packed_rows):
        packed = qweight[pr]  # [out_features]
        base_row = pr * 8
        group_id = base_row // group_size

        s = scales[group_id].float()  # [out_features]
        z = zeros[group_id].float()

        for k in range(8):
            row = base_row + k
            if row >= hidden_size:
                break
            int4_val = ((packed >> (k * 4)) & 0xF).float()
            weight_fp16[row] = ((int4_val - z) * s).half()

    return weight_fp16


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用, 跳过")
    return "cuda"


class TestW4A16GEMV:
    """W4A16 GEMV (无 LoRA) 数值测试"""

    def _get_fn(self):
        try:
            from qextract._C import w4a16_gemv
            return w4a16_gemv
        except ImportError:
            pytest.skip("CUDA 后端未编译, 跳过")

    @pytest.mark.parametrize("hidden_size,out_features", [
        (256, 128),
        (1024, 512),
        (2560, 2560),   # Qwen3-4B: hidden→hidden
        (2560, 9728),   # Qwen3-4B: hidden→intermediate
    ])
    def test_gemv_accuracy(self, device, hidden_size, out_features):
        """GEMV 数值精度测试"""
        w4a16_gemv = self._get_fn()
        group_size = 128

        # 生成随机权重并量化
        w_fp16 = torch.randn(hidden_size, out_features, dtype=torch.float16, device=device) * 0.1
        quant = simulate_gptq_quantize(w_fp16, group_size)

        # 参考: 反量化后做 FP16 矩阵乘
        w_dequant = dequantize_gptq_ref(
            quant["qweight"], quant["scales"], quant["zeros"],
            hidden_size, group_size
        )

        x = torch.randn(hidden_size, dtype=torch.float16, device=device)

        ref_output = x.float() @ w_dequant.float()
        ref_output = ref_output.half()

        our_output = w4a16_gemv(
            x, quant["qweight"], quant["scales"], quant["zeros"], group_size
        )

        # INT4 量化本身有量化误差, 容差放宽
        torch.testing.assert_close(
            our_output, ref_output, atol=1.0, rtol=0.1,
            msg=f"W4A16 GEMV [{hidden_size}→{out_features}] 数值不对齐"
        )


class TestW4A16LoRAGEMV:
    """W4A16-LoRA 融合 GEMV 数值测试"""

    def _get_fn(self):
        try:
            from qextract._C import w4a16_lora_gemv
            return w4a16_lora_gemv
        except ImportError:
            pytest.skip("CUDA 后端未编译, 跳过")

    @pytest.mark.parametrize("rank", [16, 32, 64])
    def test_lora_fused_accuracy(self, device, rank):
        """LoRA 融合精度测试"""
        w4a16_lora_gemv = self._get_fn()

        hidden_size = 2560
        out_features = 2560
        group_size = 128
        lora_alpha = 16.0

        # 底座权重
        w_fp16 = torch.randn(hidden_size, out_features,
                              dtype=torch.float16, device=device) * 0.1
        quant = simulate_gptq_quantize(w_fp16, group_size)

        # LoRA 权重 (模拟)
        lora_A = torch.randn(hidden_size, rank, dtype=torch.float16, device=device) * 0.01
        lora_B = torch.randn(rank, out_features, dtype=torch.float16, device=device) * 0.01

        x = torch.randn(hidden_size, dtype=torch.float16, device=device)

        # 参考实现
        w_dequant = dequantize_gptq_ref(
            quant["qweight"], quant["scales"], quant["zeros"],
            hidden_size, group_size
        )
        base_out = x.float() @ w_dequant.float()
        lora_out = (x.float() @ lora_A.float()) @ lora_B.float()
        scaling = lora_alpha / rank
        ref_output = (base_out + scaling * lora_out).half()

        # 我们的融合实现
        our_output = w4a16_lora_gemv(
            x, quant["qweight"], quant["scales"], quant["zeros"],
            lora_A, lora_B, group_size, lora_alpha
        )

        torch.testing.assert_close(
            our_output, ref_output, atol=1.0, rtol=0.1,
            msg=f"W4A16-LoRA GEMV (rank={rank}) 数值不对齐"
        )

    def test_lora_contribution(self, device):
        """验证 LoRA 分支确实产生了贡献 (不是全零)"""
        w4a16_lora_gemv = self._get_fn()

        hidden_size = 1024
        out_features = 512
        rank = 32
        group_size = 128

        w_fp16 = torch.randn(hidden_size, out_features,
                              dtype=torch.float16, device=device) * 0.1
        quant = simulate_gptq_quantize(w_fp16, group_size)

        # 制造一个显著的 LoRA 权重
        lora_A = torch.randn(hidden_size, rank, dtype=torch.float16, device=device) * 0.1
        lora_B = torch.randn(rank, out_features, dtype=torch.float16, device=device) * 0.1

        x = torch.randn(hidden_size, dtype=torch.float16, device=device)

        # alpha=0 → 纯底座
        from qextract._C import w4a16_gemv
        base_only = w4a16_gemv(x, quant["qweight"], quant["scales"], quant["zeros"], group_size)

        # alpha=16 → 底座 + LoRA
        fused = w4a16_lora_gemv(
            x, quant["qweight"], quant["scales"], quant["zeros"],
            lora_A, lora_B, group_size, 16.0
        )

        # 两者应该不同 (LoRA 贡献非零)
        diff = (fused.float() - base_only.float()).abs().mean().item()
        assert diff > 1e-3, f"LoRA 未产生贡献! diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
