"""
QExtract-Infer: GPU 能效监控模块
通过 NVIDIA pynvml 库高频轮询 GPU 功率, 计算 Joule/Token 能效指标

实验控制方法:
1. 每次测试前 warmup, 等 GPU 达到稳态温度
2. 测量空载基线功率 P_idle
3. 有效推理能量 = 总能量 - P_idle × duration
4. 每个配置重复 10 次取均值和标准差
5. 明确标注: "Board-level Power Proxy via NVML"
"""

import time
import threading
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class EnergyReport:
    """能效测量报告"""
    total_energy_joules: float       # 总能量 (焦耳)
    net_energy_joules: float         # 净能量 (扣除空载基线后)
    avg_power_watts: float           # 平均功率 (瓦)
    peak_power_watts: float          # 峰值功率 (瓦)
    idle_power_watts: float          # 空载基线功率 (瓦)
    duration_seconds: float          # 持续时间 (秒)
    num_tokens: Optional[int] = None  # 生成的 Token 数
    energy_per_token_mj: Optional[float] = None  # 每 Token 能量 (毫焦)
    samples: list = field(default_factory=list)  # 原始功率采样数据

    def __repr__(self):
        lines = [
            "═" * 50,
            "  QExtract Energy Report (Board-level Power Proxy)",
            "═" * 50,
            f"  Duration:        {self.duration_seconds:.3f} s",
            f"  Avg Power:       {self.avg_power_watts:.1f} W",
            f"  Peak Power:      {self.peak_power_watts:.1f} W",
            f"  Idle Baseline:   {self.idle_power_watts:.1f} W",
            f"  Total Energy:    {self.total_energy_joules:.3f} J",
            f"  Net Energy:      {self.net_energy_joules:.3f} J",
        ]
        if self.num_tokens is not None and self.energy_per_token_mj is not None:
            lines.extend([
                f"  Tokens:          {self.num_tokens}",
                f"  Energy/Token:    {self.energy_per_token_mj:.2f} mJ/token",
            ])
        lines.append("═" * 50)
        return "\n".join(lines)


class EnergyMonitor:
    """
    GPU 能效监控器

    用法:
        monitor = EnergyMonitor(device_id=0, poll_interval_ms=10)

        # 1. 先测空载基线
        monitor.calibrate_idle(duration_s=3.0)

        # 2. 开始计量
        monitor.start()
        # ... 执行推理 ...
        report = monitor.stop(num_tokens=42)

        print(report)
        print(f"Energy per token: {report.energy_per_token_mj:.2f} mJ")
    """

    def __init__(self, device_id: int = 0, poll_interval_ms: int = 10):
        try:
            import pynvml
            self._nvml = pynvml
        except ImportError:
            raise ImportError(
                "pynvml 未安装! 请运行: pip install pynvml\n"
                "或: pip install nvidia-ml-py"
            )

        self._nvml.nvmlInit()
        self.handle = self._nvml.nvmlDeviceGetHandleByIndex(device_id)
        self.poll_interval = poll_interval_ms / 1000.0  # 转为秒
        self.idle_power_watts = 0.0

        # 获取 GPU 信息
        name = self._nvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        print(f"[QExtract Energy] 监控设备: {name}")
        print(f"[QExtract Energy] 轮询间隔: {poll_interval_ms} ms")

        self._running = False
        self._samples = []
        self._thread = None

    def calibrate_idle(self, duration_s: float = 3.0):
        """
        校准空载基线功率

        在 GPU 无负载时采样 duration_s 秒, 取平均值作为 P_idle
        """
        import torch
        torch.cuda.synchronize()  # 确保之前的操作完成
        time.sleep(0.5)  # 等待 GPU 进入空闲状态

        samples = []
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            power_mw = self._nvml.nvmlDeviceGetPowerUsage(self.handle)
            samples.append(power_mw / 1000.0)  # mW → W
            time.sleep(self.poll_interval)

        if samples:
            self.idle_power_watts = sum(samples) / len(samples)
            print(f"[QExtract Energy] 空载基线: {self.idle_power_watts:.1f} W "
                  f"({len(samples)} 采样)")
        else:
            print("[QExtract Energy] ⚠️ 空载校准无采样数据")

    def start(self):
        """开始能量计量"""
        if self._running:
            raise RuntimeError("监控已在运行中")

        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, num_tokens: Optional[int] = None) -> EnergyReport:
        """
        停止计量并返回能效报告

        参数:
            num_tokens: 本次推理生成的 Token 数量 (用于计算 Energy/Token)
        """
        if not self._running:
            raise RuntimeError("监控未在运行")

        self._running = False
        self._thread.join(timeout=2.0)

        if len(self._samples) < 2:
            raise RuntimeError("采样数据不足 (至少需要 2 个采样点)")

        # ── 梯形积分计算总能量 ──
        total_energy = 0.0
        peak_power = 0.0
        power_sum = 0.0

        for i in range(1, len(self._samples)):
            t0, p0 = self._samples[i - 1]
            t1, p1 = self._samples[i]
            dt = t1 - t0
            # 梯形法: E = (p0 + p1) / 2 * dt
            total_energy += (p0 + p1) / 2.0 * dt
            peak_power = max(peak_power, p0, p1)
            power_sum += p0

        power_sum += self._samples[-1][1]
        avg_power = power_sum / len(self._samples)

        duration = self._samples[-1][0] - self._samples[0][0]
        net_energy = total_energy - self.idle_power_watts * duration
        net_energy = max(net_energy, 0.0)  # 不应为负

        # 计算 Energy per Token
        energy_per_token_mj = None
        if num_tokens is not None and num_tokens > 0:
            energy_per_token_mj = (net_energy / num_tokens) * 1000.0  # J → mJ

        return EnergyReport(
            total_energy_joules=total_energy,
            net_energy_joules=net_energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            idle_power_watts=self.idle_power_watts,
            duration_seconds=duration,
            num_tokens=num_tokens,
            energy_per_token_mj=energy_per_token_mj,
            samples=[(t, p) for t, p in self._samples],
        )

    def _poll_loop(self):
        """后台轮询线程"""
        while self._running:
            try:
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
                self._samples.append((time.perf_counter(), power_w))
            except Exception:
                pass  # 轮询失败时跳过, 不中断
            time.sleep(self.poll_interval)

    def __del__(self):
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
