"""
QExtract-Infer: End-to-End Performance Evaluator
运行真实的系统级生成测试，测量长文本任务的 TTFT (Time-To-First-Token) 和 ITL (Inter-Token Latency)
"""

import time
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from dataset_loader import IEDatasetLoader

class LatencyStreamer(BaseStreamer):
    """用于劫持 HuggingFace 生成流水线，精确记录 Prefill 和 Decode 阶段的时间戳"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.is_prompt = True
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.last_token_time = 0.0
        self.decode_intervals = []
        
    def put(self, value):
        now = time.perf_counter()
        # HuggingFace 的 streamer 在第一次调用 put 时通常会传入整个 prompt
        if self.is_prompt:
            self.is_prompt = False
            self.start_time = now
            self.last_token_time = now
        else:
            if self.first_token_time == 0.0:
                self.first_token_time = now
                self.last_token_time = now
            else:
                self.decode_intervals.append(now - self.last_token_time)
                self.last_token_time = now
                
    def end(self):
        pass

def bench_e2e(model_id, use_qextract, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Environment] 加载模型: {model_id} (Device: {device})")
    print(f"[Environment] 扩展启用状态: {'✅ QExtract' if use_qextract else '❌ PyTorch 原生'}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    if use_qextract:
        import qextract
        print("\n[Patching] 正在向模型中注入 QExtract 优化内核...")
        qextract.patch_qwen(model)

    loader = IEDatasetLoader(max_samples=num_samples)
    datasets = loader.load_all()

    streamer = LatencyStreamer()
    
    print("\n" + "="*80)
    print(" 开始端到端评测 (TTFT / ITL)".center(80))
    print("="*80)

    results_summary = []

    for dataset_name, samples in datasets.items():
        if not samples:
            continue
            
        print(f"\n>> 测试数据集: {dataset_name} ({len(samples)} 条样本)")
        ttft_list = []
        itl_list = []
        throughput_list = []
        
        for i, sample in enumerate(samples):
            prompt = sample["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs["input_ids"].shape[1]
            
            # 为了防止由于 OOM 触发导致程序崩溃，限制输入的最大长度
            if input_length > 4096:
                inputs["input_ids"] = inputs["input_ids"][:, :4096]
                inputs["attention_mask"] = inputs["attention_mask"][:, :4096]
                input_length = 4096

            streamer.reset()
            # 记录整体生成调用开始，因为 HF 有时直到算完首个 token 才调用 put
            call_start = time.perf_counter()
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64, # IE 任务 decode 极短
                    streamer=streamer,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
                
            # 修正首字延时 (TTFT): 从实际调用模型到输出第一个新 token
            if streamer.first_token_time > 0:
                ttft_ms = (streamer.first_token_time - call_start) * 1000
            else:
                ttft_ms = (time.perf_counter() - call_start) * 1000
                
            decode_len = output.shape[1] - input_length
            avg_itl_ms = (np.mean(streamer.decode_intervals) * 1000) if streamer.decode_intervals else 0.0
            
            # 总体吞吐量 (Prompt Token 解析速度)
            prefill_throughput = input_length / (ttft_ms / 1000.0) if ttft_ms > 0 else 0
            
            ttft_list.append(ttft_ms)
            itl_list.append(avg_itl_ms)
            throughput_list.append(prefill_throughput)
            
            print(f"  [{i+1}/{len(samples)}] SeqLen: {input_length:4d} | "
                  f"TTFT: {ttft_ms:6.1f} ms | ITL: {avg_itl_ms:5.1f} ms | "
                  f"Prefill: {prefill_throughput:6.1f} tok/s | Decode: {decode_len} toks")

        mean_ttft = np.mean(ttft_list)
        mean_itl = np.mean(itl_list)
        mean_thru = np.mean(throughput_list)
        results_summary.append((dataset_name, mean_ttft, mean_itl, mean_thru))

    print("\n" + "="*80)
    print(" 性能总结报告".center(80))
    print("="*80)
    print(f"| 数据集       | 首字延迟 (TTFT, ms) | 词间延迟 (ITL, ms) | Prefill 吞吐 (Token/s) |")
    print(f"|--------------|--------------------|--------------------|------------------------|")
    for row in results_summary:
        print(f"| {row[0]:<12} | {row[1]:18.1f} | {row[2]:18.1f} | {row[3]:22.1f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-GPTQ-Int4")
    parser.add_argument("--qextract", action="store_true", help="启用 QExtract 内核替换")
    parser.add_argument("--samples", type=int, default=5, help="每个数据集测试样本数")
    args = parser.parse_args()
    
    bench_e2e(args.model, args.qextract, args.samples)
