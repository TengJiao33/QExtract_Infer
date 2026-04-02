"""
QExtract-Infer 构建脚本
使用 PyTorch 的 CUDAExtension 编译自定义 CUDA 算子
用法: pip install -e .
"""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ─── 编译参数 ───
# SM 8.9 = Ada Lovelace (RTX 4060/4070/4080/4090)
# SM 8.6 = Ampere (RTX 3060/3070/3080/3090) — 向后兼容
NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_86,code=sm_86",   # Ampere
    "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-allow-unsupported-compiler",           # Bypass MSVC 2025 check
]
# Windows: /utf-8 for Unicode source files with CJK comments
if os.name == "nt":
    NVCC_FLAGS.append("-Xcompiler=/utf-8")

CXX_FLAGS = ["/O2", "/std:c++17", "/utf-8"] if os.name == "nt" else ["-O3", "-std=c++17"]

# ─── 源文件收集 ───
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
sources = [
    os.path.join(csrc_dir, "binding.cpp"),
    os.path.join(csrc_dir, "kernels", "rmsnorm.cu"),
    os.path.join(csrc_dir, "kernels", "swiglu.cu"),
    os.path.join(csrc_dir, "kernels", "w4a16_lora_gemv.cu"),
    os.path.join(csrc_dir, "kv_cache", "ring_buffer.cu"),
]

include_dirs = [
    os.path.join(csrc_dir, "include"),
]

setup(
    name="qextract-infer",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="qextract._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
