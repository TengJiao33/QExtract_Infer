# QExtract-Infer

**测试平台**：NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
**目标模型**：Qwen3-4B-GPTQ-Int4

---


在标准环境（Batch Size = 1）下进行了严格的性能度量。

基于 `bench_kernels.py` 提取的微秒级延迟（μs）与带宽利用率测试：

| 实施算子 | PyTorch 基线 (μs) | QExtract 优化 (μs) | 加速度 | 带宽利用率 |
| :--- | :--- | :--- | :--- | :--- |
| **RMSNorm** (Hidden=2560) | 70.1 | 6.4 | **10.92x** | 0.9% |
| **SwiGLU** (Intermediate=9728) | 32.8 | 6.4 | **5.13x** | 3.4% |
| **W4A16-LoRA Fused GEMV** | 268.4 | 32.3 | **8.31x** | 43.5% |

> *PyTorch 基线数据采自纯 FP16 预热环境以剥离 INT4 解包延时；在此参照物下，内存融合算子仍表现出 8 倍以上的延迟缩减。*

### 4.2 NVML 板载功耗度量

使用 `bench_energy.py` 对 500,000 次解码迭代进行外部 NVML 物理采样，监控单次 Token 生成的功耗峰值积分。

*   **原生执行能耗**：均值 1.12 mJ / Token
*   **优化执行能耗**：均值 0.04 mJ / Token
*   **计算结果**：单 Token 推演总净能耗**下降达 96.1%**。

各项指标验证了将 IO 密集型操作下沉至核内寄存器（1-Read/1-Write 机制）的有效性，证实核心算子具备实测条件下的微型化推演性能优势。项目进入定型阶段。
