"""
QExtract-Infer: Joule/Token 能效基准测试

对比 PyTorch 原生推理 vs QExtract 优化推理的 GPU 能效
输出: Energy per Token (mJ/token)
"""

import torch
import time
import argparse


def bench_energy_rmsnorm(hidden_size: int, num_iters: int = 500):
    """RMSNorm 能效对比"""
    from qextract.energy import EnergyMonitor

    device = "cuda"
    x = torch.randn(1, hidden_size, dtype=torch.float16, device=device)
    w = torch.randn(hidden_size, dtype=torch.float16, device=device)

    monitor = EnergyMonitor(poll_interval_ms=5)
    monitor.calibrate_idle(duration_s=2.0)

    # PyTorch 原生
    print(f"\n[PyTorch RMSNorm] 测量 {num_iters} 次迭代...")
    torch.cuda.synchronize()

    monitor.start()
    for _ in range(num_iters):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        out = (x.float() * torch.rsqrt(variance + 1e-6) * w.float()).half()
    torch.cuda.synchronize()
    pytorch_report = monitor.stop(num_tokens=num_iters)
    print(pytorch_report)

    # QExtract 融合
    try:
        from qextract._C import fused_rmsnorm

        print(f"\n[QExtract RMSNorm] 测量 {num_iters} 次迭代...")
        torch.cuda.synchronize()
        time.sleep(1.0)  # 冷却

        monitor.calibrate_idle(duration_s=2.0)
        monitor.start()
        for _ in range(num_iters):
            out = fused_rmsnorm(x, w, 1e-6)
        torch.cuda.synchronize()
        qextract_report = monitor.stop(num_tokens=num_iters)
        print(qextract_report)

        if pytorch_report.energy_per_token_mj and qextract_report.energy_per_token_mj:
            reduction = (1 - qextract_report.energy_per_token_mj / pytorch_report.energy_per_token_mj) * 100
            print(f"\n⚡ 能效提升: {reduction:.1f}%")
            print(f"   PyTorch:  {pytorch_report.energy_per_token_mj:.2f} mJ/token")
            print(f"   QExtract: {qextract_report.energy_per_token_mj:.2f} mJ/token")

    except ImportError:
        print("⚠️ CUDA 后端未编译, 仅测试 PyTorch 基线")


def main():
    parser = argparse.ArgumentParser(description="QExtract-Infer 能效基准测试")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("  QExtract-Infer Energy Benchmark")
    print("  Board-level Power Proxy via NVML")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    bench_energy_rmsnorm(args.hidden_size, args.iters)


if __name__ == "__main__":
    main()
