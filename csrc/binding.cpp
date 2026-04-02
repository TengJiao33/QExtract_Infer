/**
 * QExtract-Infer: PyBind11 绑定入口
 * 将所有 CUDA 算子暴露为 Python 的 qextract._C 模块
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ─── 前向声明 (定义在各 .cu 文件中) ───
// RMSNorm
torch::Tensor fused_rmsnorm(torch::Tensor input, torch::Tensor weight, float eps);

// SwiGLU
torch::Tensor fused_swiglu(torch::Tensor gate, torch::Tensor up);

// W4A16 GEMV
torch::Tensor w4a16_gemv(
    torch::Tensor input, torch::Tensor qweight,
    torch::Tensor scales, torch::Tensor zeros, int group_size);

// W4A16-LoRA Fused GEMV
torch::Tensor w4a16_lora_gemv(
    torch::Tensor input, torch::Tensor qweight,
    torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor lora_A, torch::Tensor lora_B,
    int group_size, float lora_alpha);

// ─── Ring Buffer KV Cache (前向声明类) ───
// 类定义在 ring_buffer.cu 中, 通过 pybind11 暴露
class RingBufferKVCache;

// ─── PyBind11 模块定义 ───
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "QExtract-Infer: 面向长文本信息抽取的 QLoRA 融合推理微内核";

    // 融合算子
    m.def("fused_rmsnorm", &fused_rmsnorm,
          "融合 RMSNorm (1读1写, Warp Shuffle 归约)",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6f);

    m.def("fused_swiglu", &fused_swiglu,
          "融合 SwiGLU 激活函数 (gate * sigmoid(gate) * up)",
          py::arg("gate"), py::arg("up"));

    m.def("w4a16_gemv", &w4a16_gemv,
          "W4A16 GEMV: INT4 底座权重在线反量化 + 向量点积",
          py::arg("input"), py::arg("qweight"),
          py::arg("scales"), py::arg("zeros"),
          py::arg("group_size") = 128);

    m.def("w4a16_lora_gemv", &w4a16_lora_gemv,
          "W4A16-LoRA 融合 GEMV: 底座反量化 + LoRA 补偿一体化",
          py::arg("input"), py::arg("qweight"),
          py::arg("scales"), py::arg("zeros"),
          py::arg("lora_A"), py::arg("lora_B"),
          py::arg("group_size") = 128,
          py::arg("lora_alpha") = 16.0f);

    // Ring Buffer KV Cache
    py::class_<RingBufferKVCache>(m, "RingBufferKVCache",
        "StreamingLLM 风格的静态连续环形 KV-Cache")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_layers"),
             py::arg("num_kv_heads"),
             py::arg("head_dim"),
             py::arg("sink_size") = 4,
             py::arg("window_size") = 4092)
        .def("append", &RingBufferKVCache::append,
             "追加新 Token 的 KV 到指定层",
             py::arg("new_k"), py::arg("new_v"), py::arg("layer_idx"))
        .def("get_kv", &RingBufferKVCache::get_kv,
             "获取指定层的有效 KV 序列 (sink + ring)")
        .def("get_valid_length", &RingBufferKVCache::get_valid_length)
        .def("get_total_written", &RingBufferKVCache::get_total_written)
        .def("reset", &RingBufferKVCache::reset);
}
