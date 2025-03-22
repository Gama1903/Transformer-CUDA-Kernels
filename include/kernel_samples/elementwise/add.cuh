/**
 * @file add.cuh
 * @author Gama1903 (gama1903@qq.com)
 * @brief 亮点：相比原文件，提供统一的模板接口和错误处理
 * @version 0.1
 * @date 2025-03-21
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_fp16.h>
#include <kernel_samples/base/common.cuh>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

void add_cpu(float *a, float *b, float *c, int const N)
{
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// -------------------------------------- FP32 --------------------------------------

__global__ void add_f32_kernel(float *a, float *b, float *c, int const N)
{
    int const global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_tid < N)
        c[global_tid] = a[global_tid] + b[global_tid];
}

// N % 4 == 0
__global__ void add_f32x4_kernel(float *a, float *b, float *c, int const N)
{
    int const global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int const vec_base_idx = global_tid * 4;

    if (vec_base_idx < N)
    {
        // load
        float4 reg_a = FLOAT4(a[vec_base_idx]);
        float4 reg_b = FLOAT4(b[vec_base_idx]);

        // compute
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        // store
        FLOAT4(c[vec_base_idx]) = reg_c;
    }
}

// VEC_LEN = 1, 4
template <int VEC_LEN>
void add_f32(float *a, float *b, float *c, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 4)
        add_f32x4_kernel<<<grid_size, BLOCK_SIZE>>>(a, b, c, N);
    else if constexpr (VEC_LEN == 1)
        add_f32_kernel<<<grid_size, BLOCK_SIZE>>>(a, b, c, N);
    else
        static_assert(VEC_LEN != VEC_LEN, "Invalid VEC_LEN");
}

// -------------------------------------- FP16 --------------------------------------

__global__ void add_f16_kernel(half *a, half *b, half *c, int const N)
{
    int const global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_tid < N)
        c[global_tid] = __hadd(a[global_tid], b[global_tid]);
}

// N % 8 == 0
__global__ void add_f16x8_kernel(half *a, half *b, half *c, int const N)
{
    int const global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int const vec_base_idx = global_tid * 8;
    if (vec_base_idx < N)
    {
        half reg_a[8], reg_b[8], reg_c[8];

        // load
        LDST128BITS(reg_a[0]) = LDST128BITS(a[vec_base_idx]);
        LDST128BITS(reg_b[0]) = LDST128BITS(b[vec_base_idx]);

        // compute
#pragma unroll
        for (int offset = 0; offset < 8; offset += 2)
        {
            HALF2(reg_c[offset]) = __hadd2(HALF2(reg_a[offset]), HALF2(reg_b[offset]));
        }

        // store
        LDST128BITS(c[vec_base_idx]) = LDST128BITS(reg_c[0]);
    }
}

// VEC_LEN = 1, 8
template <int VEC_LEN>
void add_f16(half *a, half *b, half *c, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 8)
        add_f16x8_kernel<<<grid_size, BLOCK_SIZE>>>(a, b, c, N);
    else if constexpr (VEC_LEN == 1)
        add_f16_kernel<<<grid_size, BLOCK_SIZE>>>(a, b, c, N);
    else
        static_assert(VEC_LEN != VEC_LEN, "Invalid VEC_LEN");
}

// -------------------------------------- API --------------------------------------

template <class Tp, int VEC_LEN>
void add(Tp *a, Tp *b, Tp *c, int const N)
{
    if constexpr (std::is_same_v<Tp, float>)
    {
        if constexpr (VEC_LEN == 0)
            add_cpu(a, b, c, N);
        else
            add_f32<VEC_LEN>(a, b, c, N);
    }
    else if constexpr (std::is_same_v<Tp, half>)
        add_f16<VEC_LEN>(a, b, c, N);
    else
        static_assert(!std::is_same_v<Tp, Tp>, "Unsupported type");
}

// -------------------------------------- END --------------------------------------