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
        from qextract._C import fused_rmsnorm, fused_swiglu, w4a16_lora_gemv
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

        # ── 替换 LoRA GEMV ──
        if enable_lora_gemv:
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                for sub_module in [getattr(layer, "self_attn", None), getattr(layer, "mlp", None)]:
                    if sub_module is not None:
                        proj_module = getattr(sub_module, proj_name, None)
                        if proj_module is not None:
                            if _patch_lora_gemv(proj_module, w4a16_lora_gemv):
                                patched["lora_gemv"] += 1

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


def _patch_lora_gemv(proj_module, fused_fn):
    """提取 PEFT 和 GPTQ 权重，替换包含 LoRA 适配器的 Linear 处理"""
    if not hasattr(proj_module, "base_layer"):
        return False
        
    base_layer = proj_module.base_layer
    
    # 检查是否为 GPTQ 量化层
    if not hasattr(base_layer, "qweight") or not hasattr(base_layer, "scales"):
        return False
        
    # 提取量化底座权重
    qweight = base_layer.qweight
    scales = base_layer.scales
    zeros = getattr(base_layer, "qzeros", getattr(base_layer, "zeros", None))
    if zeros is None:
        return False
        
    # 获取组大小
    group_size = getattr(base_layer, "group_size", 128)
    
    # 检查是否有 'default' LoRA 适配器
    if not hasattr(proj_module, "lora_A") or "default" not in proj_module.lora_A:
        return False
        
    # 提取并转置 LoRA 参数:
    # PEFT 默认存储 nn.Linear 权重:
    # lora_A.weight 形状是 [rank, in_features] -> 转置后为 [in_features, rank]
    # lora_B.weight 形状是 [out_features, rank] -> 转置后为 [rank, out_features]
    lora_A_t = proj_module.lora_A["default"].weight.detach().t().contiguous().half()
    lora_B_t = proj_module.lora_B["default"].weight.detach().t().contiguous().half()
    lora_alpha = getattr(proj_module, "lora_alpha", {}).get("default", 16.0)

    def new_forward(self, x):
        input_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.half()
            
        fused_output = fused_fn(
            x, qweight, scales, zeros,
            lora_A_t, lora_B_t, group_size, lora_alpha
        )
        return fused_output.to(input_dtype)

    proj_module.forward = types.MethodType(new_forward, proj_module)
    return True
