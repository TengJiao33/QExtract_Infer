#pragma once

#include "common.h"

// ─── INT4 量化工具函数 ───
// GPTQ 格式：每 8 个 INT4 权重打包为 1 个 INT32
// 布局：packed_int32 = w0 | (w1 << 4) | (w2 << 8) | ... | (w7 << 28)
// 反量化：fp16_weight = (int4_value - zero_point) * scale

// 从一个 uint32 中解包 8 个 INT4 值并反量化为 FP16
// packed: 打包的 INT4 权重
// out: 输出 8 个 half 值
// scale: 该 group 的缩放因子
// zero_point: 该 group 的零点
__device__ __forceinline__ void dequant_int4x8(
    uint32_t packed,
    half_t* out,
    half_t scale,
    half_t zero_point
) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // 提取第 i 个 4-bit 值 (无符号, 0~15)
        int int4_val = (packed >> (i * 4)) & 0xF;
        // 反量化: (value - zero_point) * scale
        float dequant = (__half2float(scale)) *
                        (static_cast<float>(int4_val) - __half2float(zero_point));
        out[i] = __float2half(dequant);
    }
}

// 高效版本：利用 half2 运算一次处理 2 个值
__device__ __forceinline__ void dequant_int4x8_fast(
    uint32_t packed,
    half2_t* out,     // 输出 4 个 half2 = 8 个 half
    float scale_f,
    float zp_f
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v0 = (packed >> (i * 8))     & 0xF;
        int v1 = (packed >> (i * 8 + 4)) & 0xF;
        float f0 = scale_f * (static_cast<float>(v0) - zp_f);
        float f1 = scale_f * (static_cast<float>(v1) - zp_f);
        out[i] = __floats2half2_rn(f0, f1);
    }
}

// ─── LoRA 相关常量 ───
// 典型 LoRA rank 范围
constexpr int LORA_RANK_MIN = 8;
constexpr int LORA_RANK_MAX = 64;
constexpr int GPTQ_GROUP_SIZE = 128;  // 默认分组大小
