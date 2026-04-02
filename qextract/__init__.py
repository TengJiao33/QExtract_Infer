"""
QExtract-Infer: 面向长文本信息抽取的 QLoRA 融合推理微内核

用法:
    import qextract
    qextract.patch_qwen(model)  # 一键替换底层算子
"""

__version__ = "0.1.0"

try:
    from qextract._C import (
        fused_rmsnorm,
        fused_swiglu,
        w4a16_gemv,
        w4a16_lora_gemv,
        RingBufferKVCache,
    )
    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False

from qextract.patch import patch_qwen
from qextract.energy import EnergyMonitor


def check_backend():
    """检查 CUDA 后端是否已编译"""
    if _C_AVAILABLE:
        print("[QExtract] ✅ CUDA 后端已加载")
    else:
        print("[QExtract] ⚠️ CUDA 后端未编译, 请运行: pip install -e .")
    return _C_AVAILABLE
