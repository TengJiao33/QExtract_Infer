"""
QExtract-Infer: 单算子性能基准测试

测量每个 CUDA 算子的:
- 延迟 (μs)
- 吞吐量 (GB/s)
- 显存带宽利用率 (% of peak)
- 与 PyTorch 原生实现的加速比

RTX 4060 理论显存带宽: 272 GB/s
"""

import torch
import time
import argparse
from typing import Callable


PEAK_BANDWIDTH_GBS = 272.0  # RTX 4060 理论峰值
WARMUP_ITERS = 50
BENCH_ITERS = 200


def bench_kernel(
    fn: Callable,
    args: tuple,
    label: str,
    bytes_accessed: int,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> dict:
    """通用 Kernel 基准测试"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    elapsed_us = elapsed_ms * 1000

    throughput_gbs = (bytes_accessed / 1e9) / (elapsed_ms / 1e3)
    utilization = throughput_gbs / PEAK_BANDWIDTH_GBS * 100

    return {
        "label": label,
        "latency_us": elapsed_us,
        "throughput_gbs": throughput_gbs,
        "utilization_pct": utilization,
        "bytes_accessed": bytes_accessed,
    }


def rmsnorm_pytorch_ref(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).half()


def swiglu_pytorch_ref(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).half()


def w4a16_lora_gemv_pytorch_ref(x, w_fp16, lora_A, lora_B, scaling):
    base_out = torch.matmul(x.float(), w_fp16.float())
    lora_out = torch.matmul(torch.matmul(x.float(), lora_A.float()), lora_B.float())
    return (base_out + scaling * lora_out).half()


def bench_rmsnorm(hidden_size: int, batch_size: int = 1):
    """RMSNorm 基准测试"""
    device = "cuda"
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
    w = torch.randn(hidden_size, dtype=torch.float16, device=device)

    # 数据量: 读 x + 读 weight + 写 output = 2*batch*hidden + hidden (half)
    bytes_accessed = (2 * batch_size * hidden_size + hidden_size) * 2

    # PyTorch 原生
    pytorch_result = bench_kernel(
        lambda: rmsnorm_pytorch_ref(x, w),
        (),
        f"PyTorch RMSNorm ({hidden_size})",
        bytes_accessed,
    )

    # QExtract 融合
    try:
        from qextract._C import fused_rmsnorm
        qextract_result = bench_kernel(
            lambda: fused_rmsnorm(x, w, 1e-6),
            (),
            f"QExtract RMSNorm ({hidden_size})",
            bytes_accessed,
        )
        speedup = pytorch_result["latency_us"] / qextract_result["latency_us"]
        qextract_result["speedup"] = speedup
    except ImportError:
        qextract_result = None

    return pytorch_result, qextract_result


def bench_swiglu(intermediate_size: int, batch_size: int = 1):
    """SwiGLU 基准测试"""
    device = "cuda"
    gate = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)
    up = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device=device)

    # 数据量: 读 gate + 读 up + 写 output
    bytes_accessed = 3 * batch_size * intermediate_size * 2

    pytorch_result = bench_kernel(
        lambda: swiglu_pytorch_ref(gate, up),
        (),
        f"PyTorch SwiGLU ({intermediate_size})",
        bytes_accessed,
    )

    try:
        from qextract._C import fused_swiglu
        qextract_result = bench_kernel(
            lambda: fused_swiglu(gate, up),
            (),
            f"QExtract SwiGLU ({intermediate_size})",
            bytes_accessed,
        )
        speedup = pytorch_result["latency_us"] / qextract_result["latency_us"]
        qextract_result["speedup"] = speedup
    except ImportError:
        qextract_result = None

    return pytorch_result, qextract_result


def bench_lora_fused_gemv(hidden_size: int, out_features: int, batch_size: int = 1):
    """W4A16-LoRA Fused GEMV 基准测试"""
    rank = 32
    group_size = 128
    device = "cuda"
    
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
    w_fp16 = torch.randn(hidden_size, out_features, dtype=torch.float16, device=device)
    
    lora_A = torch.randn(hidden_size, rank, dtype=torch.float16, device=device)
    lora_B = torch.randn(rank, out_features, dtype=torch.float16, device=device)
    lora_alpha = 16.0
    scaling = lora_alpha / rank
    
    # 动态加载测试用的 GPTQ 模拟函数 (由于 benchmarks 目录和 tests 目录平行，需处理路径或直接 sys.path.append)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tests.test_w4a16_gemv import simulate_gptq_quantize
    
    quant = simulate_gptq_quantize(w_fp16, group_size)
    qweight, scales, zeros = quant["qweight"], quant["scales"], quant["zeros"]

    # 访存量分析：
    # INT4 权重: (hidden * out) / 2 bytes
    # scales/zeros: 2 * (hidden / group) * out * (2 bytes float16)  (*2是由于有scales和zeros)
    # LoRA_A/B: (hidden*rank + rank*out) * 2 bytes
    # 读 x, 写 out: (batch*hidden + batch*out) * 2 bytes
    bytes_accessed = (
        (hidden_size * out_features) // 2 +
        2 * (hidden_size // group_size) * out_features * 2 +
        (hidden_size * rank + rank * out_features) * 2 +
        (batch_size * hidden_size + batch_size * out_features) * 2
    )

    # PyTorch Baseline 我们用纯 FP16 乘法，代表最理想的带宽情况（不需要 Dequant CPU/GPU overhead）
    pytorch_result = bench_kernel(
        lambda: w4a16_lora_gemv_pytorch_ref(x, w_fp16, lora_A, lora_B, scaling),
        (),
        f"PyTorch FP16+LoRA ({hidden_size}x{out_features})",
        bytes_accessed,
    )

    try:
        from qextract._C import w4a16_lora_gemv
        qextract_result = bench_kernel(
            lambda: w4a16_lora_gemv(x, qweight, scales, zeros, lora_A, lora_B, group_size, lora_alpha),
            (),
            f"QExtract W4A16-LoRA ({hidden_size}x{out_features})",
            bytes_accessed,
        )
        speedup = pytorch_result["latency_us"] / qextract_result["latency_us"]
        qextract_result["speedup"] = speedup
    except ImportError:
        qextract_result = None

    return pytorch_result, qextract_result


def print_results(results: list):
    """以 Markdown 表格格式打印结果"""
    print()
    print("| Kernel | Latency (μs) | Throughput (GB/s) | BW Util (%) | Speedup |")
    print("|--------|-------------|-------------------|-------------|---------|")

    for r in results:
        if r is None:
            continue
        speedup = f"{r.get('speedup', '-'):.2f}x" if isinstance(r.get('speedup'), float) else "-"
        print(f"| {r['label']:<40} | {r['latency_us']:>11.1f} | {r['throughput_gbs']:>17.1f} | {r['utilization_pct']:>11.1f} | {speedup:>7} |")

    print()


def main():
    parser = argparse.ArgumentParser(description="QExtract-Infer 算子基准测试")
    parser.add_argument("--model", choices=["qwen3-4b", "qwen3-8b"], default="qwen3-4b")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    configs = {
        "qwen3-4b": {"hidden": 2560, "intermediate": 9728},
        "qwen3-8b": {"hidden": 4096, "intermediate": 12288},
    }
    cfg = configs[args.model]

    print(f"\n{'='*60}")
    print(f"  QExtract-Infer 算子基准测试")
    print(f"  模型: {args.model}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  峰值带宽: {PEAK_BANDWIDTH_GBS} GB/s")
    print(f"  Batch Size: {args.batch}")
    print(f"{'='*60}")

    all_results = []

    # RMSNorm
    print("\n[1/3] 测试 RMSNorm...")
    pt, qe = bench_rmsnorm(cfg["hidden"], args.batch)
    all_results.extend([pt, qe])

    # SwiGLU
    print("[2/3] 测试 SwiGLU...")
    pt, qe = bench_swiglu(cfg["intermediate"], args.batch)
    all_results.extend([pt, qe])

    # W4A16-LoRA GEMV
    print("[3/3] 测试 W4A16-LoRA Fused GEMV...")
    pt, qe = bench_lora_fused_gemv(cfg["hidden"], cfg["hidden"], args.batch)
    all_results.extend([pt, qe])

    print_results([r for r in all_results if r is not None])


if __name__ == "__main__":
    main()
