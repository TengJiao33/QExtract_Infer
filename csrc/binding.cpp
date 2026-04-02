/**
 * QExtract-Infer: PyBind11 Binding Entry
 * Expose all CUDA operators as Python qextract._C module
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ring_buffer.h"

// Forward declarations (defined in .cu files)
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

// PyBind11 Module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "QExtract-Infer: QLoRA Fused Inference Micro-Kernel for IE Tasks";

    // Fused operators
    m.def("fused_rmsnorm", &fused_rmsnorm,
          "Fused RMSNorm (1R+1W, Warp Shuffle reduction)",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6f);

    m.def("fused_swiglu", &fused_swiglu,
          "Fused SwiGLU activation (gate * sigmoid(gate) * up)",
          py::arg("gate"), py::arg("up"));

    m.def("w4a16_gemv", &w4a16_gemv,
          "W4A16 GEMV: INT4 base weight online dequant + vector dot product",
          py::arg("input"), py::arg("qweight"),
          py::arg("scales"), py::arg("zeros"),
          py::arg("group_size") = 128);

    m.def("w4a16_lora_gemv", &w4a16_lora_gemv,
          "W4A16-LoRA Fused GEMV: base dequant + LoRA compensation fused",
          py::arg("input"), py::arg("qweight"),
          py::arg("scales"), py::arg("zeros"),
          py::arg("lora_A"), py::arg("lora_B"),
          py::arg("group_size") = 128,
          py::arg("lora_alpha") = 16.0f);

    // Ring Buffer KV Cache
    py::class_<RingBufferKVCache>(m, "RingBufferKVCache",
        "StreamingLLM-style static contiguous ring KV-Cache")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_layers"),
             py::arg("num_kv_heads"),
             py::arg("head_dim"),
             py::arg("sink_size") = 4,
             py::arg("window_size") = 4092)
        .def("append", &RingBufferKVCache::append,
             "Append new token KV to specified layer",
             py::arg("new_k"), py::arg("new_v"), py::arg("layer_idx"))
        .def("get_kv", &RingBufferKVCache::get_kv,
             "Get valid KV sequence for specified layer (sink + ring)")
        .def("get_valid_length", &RingBufferKVCache::get_valid_length)
        .def("get_total_written", &RingBufferKVCache::get_total_written)
        .def("reset", &RingBufferKVCache::reset);
}
