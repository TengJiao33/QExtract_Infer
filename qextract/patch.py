"""
QExtract-Infer: Monkey Patching 模块
一键将优化算子注入 HuggingFace Qwen3 模型
"""

import torch
import types


def patch_qwen(model, enable_rmsnorm=True, enable_swiglu=True, enable_lora_gemv=True):
    """
    一键替换 Qwen3 模型中的底层算子

    用法:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-GPTQ-Int4")
        import qextract
        qextract.patch_qwen(model)

    参数:
        model: HuggingFace Qwen3 模型实例
        enable_rmsnorm: 是否替换 RMSNorm
        enable_swiglu: 是否替换 SwiGLU 激活
        enable_lora_gemv: 是否替换为 W4A16-LoRA 融合 GEMV
    """
    try:
        from qextract._C import fused_rmsnorm, fused_swiglu
    except ImportError:
        raise RuntimeError(
            "[QExtract] CUDA 后端未编译！请先运行: pip install -e .\n"
            "需要: CUDA Toolkit + Visual Studio Build Tools"
        )

    patched = {"rmsnorm": 0, "swiglu": 0, "lora_gemv": 0}

    for layer in model.model.layers:
        # ── 替换 RMSNorm ──
        if enable_rmsnorm:
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                norm_module = getattr(layer, norm_name, None)
                if norm_module is not None:
                    _patch_rmsnorm(norm_module, fused_rmsnorm)
                    patched["rmsnorm"] += 1

        # ── 替换 SwiGLU 激活 ──
        if enable_swiglu and hasattr(layer, "mlp"):
            _patch_swiglu(layer.mlp, fused_swiglu)
            patched["swiglu"] += 1

    # ── 替换模型最终的 RMSNorm ──
    if enable_rmsnorm and hasattr(model.model, "norm"):
        _patch_rmsnorm(model.model.norm, fused_rmsnorm)
        patched["rmsnorm"] += 1

    print(f"[QExtract] ✅ 已注入优化算子:")
    print(f"  - RMSNorm: {patched['rmsnorm']} 处")
    print(f"  - SwiGLU:  {patched['swiglu']} 处")
    print(f"  - LoRA GEMV: {patched['lora_gemv']} 处")

    return model


def _patch_rmsnorm(norm_module, fused_fn):
    """替换单个 RMSNorm 模块的 forward 方法"""
    weight = norm_module.weight.data.half()
    eps = getattr(norm_module, "variance_epsilon",
                  getattr(norm_module, "eps", 1e-6))

    def new_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # 确保输入是 FP16
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.half()
        output = fused_fn(hidden_states, weight, eps)
        return output.to(input_dtype)

    norm_module.forward = types.MethodType(new_forward, norm_module)


def _patch_swiglu(mlp_module, fused_fn):
    """替换 MLP 模块中的 SwiGLU 激活"""
    original_forward = mlp_module.forward

    def new_forward(self, x):
        # Qwen3 MLP 结构:
        # gate = self.gate_proj(x)   # Linear
        # up = self.up_proj(x)       # Linear
        # act = self.act_fn(gate)    # SiLU — 被替换
        # output = act * up          # 逐元素乘 — 被替换
        # output = self.down_proj(output)  # Linear

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # 融合 SiLU + 逐元素乘
        if gate.dtype != torch.float16:
            gate = gate.half()
            up = up.half()
        fused_output = fused_fn(gate, up)

        output = self.down_proj(fused_output.to(x.dtype))
        return output

    mlp_module.forward = types.MethodType(new_forward, mlp_module)
