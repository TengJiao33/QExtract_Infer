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
    print("\n[1/2] 测试 RMSNorm...")
    pt, qe = bench_rmsnorm(cfg["hidden"], args.batch)
    all_results.extend([pt, qe])

    # SwiGLU
    print("[2/2] 测试 SwiGLU...")
    pt, qe = bench_swiglu(cfg["intermediate"], args.batch)
    all_results.extend([pt, qe])

    print_results([r for r in all_results if r is not None])


if __name__ == "__main__":
    main()
