# QExtract-Infer

<p align="center">
<b>面向长文本信息抽取的 QLoRA 极致融合与极简流式推理微内核</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia" alt="CUDA"/>
<img src="https://img.shields.io/badge/PyTorch-2.7-orange?logo=pytorch" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Qwen3-4B%2F8B-blue" alt="Qwen3"/>
<img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 💡 为什么不用 vLLM？

主流推理框架 (vLLM、TGI) 是为**通用聊天 (Chat)** 场景设计的——多轮对话长度不可测，必须引入 PagedAttention 动态分页。但信息抽取 (IE) 任务有着截然不同的负载特征：

| | 通用聊天 | 信息抽取 (IE) |
|---|---|---|
| 输入 (Prefill) | 中等长度 | **极长** (26万条评论) |
| 输出 (Decode) | 不定长 | **极短** (结构化三元组) |
| KV-Cache 压力 | 高 (需要动态分页) | **低** (输出短, 可预分配) |

**QExtract-Infer** 抛弃通用框架的冗余抽象，针对 IE 的"长 Prefill + 短 Decode"极端场景，从零打造极简推理内核。

---

## 🔥 核心技术亮点

### 1. W4A16-LoRA Fused GEMV — 算子融合核武器

将原生 PyTorch 的 **4 次全局显存读写压缩为 1 次**：

```
原生 PyTorch (串行, 4+ 次访存):
  dequant(W_int4) → FP16    [读写1]
  Y_base = X · W_fp16       [读写2]
  Y_lora_A = X · A          [读写3]
  Y_lora = Y_lora_A · B     [读写4]
  Y = Y_base + α · Y_lora   [读写5]

QExtract (单 Kernel, 1 次读写):
  在 Shared Memory + 寄存器层面完成
  底座反量化 → 底座乘法 → LoRA 补偿 → 合并 → 一次写回
```

### 2. StreamingLLM Ring Buffer KV-Cache

结合 Attention Sink 理论，用 C++ 裸指针在 O(1) 显存下支持长文本：

```
┌──────────────────────────────────────┐
│ Sink (前 4 个 Token, 永不覆盖)        │
├──────────────────────────────────────┤
│ Ring Buffer (循环覆盖最旧 Token)      │
└──────────────────────────────────────┘
```

### 3. Energy per Token — 硬件级能效指标

通过 NVIDIA NVML 监控 GPU 板卡功率，独创性地给出推理能耗数据：

> ⚠️ Board-level Power Proxy: 板卡级功率代理指标，非算子级精确能耗

---

## 📦 安装

### 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA)
- CUDA Toolkit 12.x
- Visual Studio 2022 Build Tools (Windows) / GCC (Linux)

### 安装步骤

```bash
git clone https://github.com/your-repo/QExtract-Infer.git
cd QExtract-Infer

# 开发模式安装 (编译 CUDA 扩展)
pip install -e .

# 验证
python -c "import qextract; qextract.check_backend()"
```

---

## 🚀 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import qextract

# 加载 GPTQ 量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-GPTQ-Int4",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-GPTQ-Int4")

# 一键注入优化算子
qextract.patch_qwen(model)

# 正常推理
inputs = tokenizer("请从以下评论中抽取情感三元组：...", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
```

---

## 📊 性能数据

> 以下数据基于 RTX 4060 Laptop GPU (8GB) 测量

### 单算子延迟 (Qwen3-4B 配置, Batch=1)

| 算子 | PyTorch (μs) | QExtract (μs) | 加速比 | BW 利用率 |
|------|-------------|---------------|--------|----------|
| RMSNorm (2560) | TBD | TBD | TBD | TBD |
| SwiGLU (9728) | TBD | TBD | TBD | TBD |
| W4A16 GEMV | TBD | TBD | TBD | TBD |
| W4A16-LoRA GEMV | TBD | TBD | TBD | TBD |

### 能效对比 (Energy per Token)

| 配置 | Energy/Token (mJ) | 功耗削减 |
|------|-------------------|---------|
| PyTorch 原生 | TBD | - |
| QExtract 优化 | TBD | TBD |

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行基准测试
python benchmarks/bench_kernels.py --model qwen3-4b

# 运行能效测试
python benchmarks/bench_energy.py
```

---

## 🏗️ 项目结构

```
QExtract/
├── csrc/                    # C++/CUDA 源码
│   ├── binding.cpp          # PyBind11 绑定
│   ├── include/
│   │   ├── common.h         # 公共宏、Warp 归约
│   │   └── quantize_utils.h # INT4 解包工具
│   ├── kernels/
│   │   ├── rmsnorm.cu       # 融合 RMSNorm
│   │   ├── swiglu.cu        # 融合 SwiGLU
│   │   └── w4a16_lora_gemv.cu  # 核武器
│   └── kv_cache/
│       └── ring_buffer.cu   # Ring Buffer KV-Cache
├── qextract/                # Python 包
│   ├── __init__.py
│   ├── patch.py             # Monkey Patching
│   ├── kv_cache.py          # KV-Cache 封装
│   └── energy.py            # 能效监控
├── tests/                   # 测试
├── benchmarks/              # 基准测试
├── setup.py                 # 构建脚本
└── README.md
```

---

## 📝 技术细节

### Qwen3 模型配置

| 参数 | Qwen3-4B | Qwen3-8B |
|------|----------|----------|
| `hidden_size` | 2560 | 4096 |
| `intermediate_size` | 9728 | 12288 |
| `num_layers` | 36 | 36 |
| `num_attention_heads` | 32 | 32 |
| `num_kv_heads` (GQA) | 8 | 8 |
| `head_dim` | 128 | 128 |

### KV-Cache 显存计算

```
每 Token = 2(K+V) × 8(KV头数) × 128(head_dim) × 36(层) × 2B(FP16)
Qwen3-4B: 56 KB/Token → 4096 Token ≈ 224 MB
Qwen3-8B: 72 KB/Token → 4096 Token ≈ 288 MB
```

---

## 📄 License

MIT License
