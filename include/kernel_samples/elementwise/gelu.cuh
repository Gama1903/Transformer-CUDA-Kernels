/**
 * @file gelu.cuh
 * @author Gama1903 (gama1903@qq.com)
 * @brief 亮点
 *        1. 修复了原文件中截断数值范围过大的问题
 *        2. 在原文件基础上，使用模板函数，给出了不同数据类型和不同向量化长度核函数的统一接口，并给出了错误处理
 * @version 0.1
 * @date 2025-03-21
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cmath>
#include <cuda_fp16.h>
#include <kernel_samples/base/common.cuh>

#define MAX_EXP_F32 10.0f
#define MIN_EXP_F32 -10.0f
#define MAX_EXP_F16 __float2half(4.0f)
#define MIN_EXP_F16 __float2half(-4.0f)

#define HALF_DIV2 __float2half(0.5f)
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_SQRT_2_PI __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2

#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f

#define GELU gelu_tanh_approximate

#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&(val))[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val))[0])

__forceinline__ __device__ float gelu_tanh_approximate(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__forceinline__ __device__ half gelu_tanh_approximate(half x)
{
    half x_cube = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + __float2half(0.044715f) * x_cube);
    return HALF_DIV2 * x * (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));
}

// -------------------------------------- FP32 --------------------------------------

__global__ void gelu_f32_kernel(float *x, float *y, int const N)
{
    int const global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < N)
    {
        float val = fminf(fmaxf(x[global_tid], MIN_EXP_F32), MAX_EXP_F32);
        y[global_tid] = GELU(val);
    }
}

// N % 4 == 0
__global__ void gelu_f32x4_kernel(float *x, float *y, int const N)
{
    int const global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const vec_base_tid = global_tid * 4;
    if (vec_base_tid < N)
    {
        // load
        float4 reg_x = FLOAT4(x[vec_base_tid]);

        // clip
        reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

        // compute
        float4 reg_y;
        reg_y.x = GELU(reg_x.x);
        reg_y.y = GELU(reg_x.y);
        reg_y.z = GELU(reg_x.z);
        reg_y.w = GELU(reg_x.w);

        // store
        FLOAT4(y[vec_base_tid]) = reg_y;
    }
}

// VEC_LEN = 1, 4
template <int VEC_LEN>
void gelu_f32(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 1)
        gelu_f32_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
    else if constexpr (VEC_LEN == 4)
        gelu_f32x4_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
    else
        static_assert(VEC_LEN != VEC_LEN, "Invalid VEC_LEN");
}

// -------------------------------------- FP16 --------------------------------------

__global__ void gelu_f16_kernel(half *x, half *y, int const N)
{
    int const global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < N)
    {
        half val = __hmin(__hmax(x[global_tid], MIN_EXP_F16), MAX_EXP_F16);
        y[global_tid] = GELU(val);
    }
}

// N % 8 == 0
__global__ void gelu_f16x8_kernel(half *x, half *y, int const N)
{
    int const global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const vec_base_tid = global_tid * 8;
    if (vec_base_tid < N)
    {
        half reg_a[8], reg_b[8];

        // load
        LDST128BITS(reg_a[0]) = LDST128BITS(x[vec_base_tid]);
        LDST128BITS(reg_b[0]) = LDST128BITS(y[vec_base_tid]);

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            // clip
            reg_a[i] = __hmin(__hmax(reg_a[i], MIN_EXP_F16), MAX_EXP_F16);

            // compute
            reg_b[i] = GELU(reg_a[i]);
        }

        // store
        LDST128BITS(y[vec_base_tid]) = LDST128BITS(reg_b[0]);
    }
}

// VEC_LEN = 1, 8
template <int VEC_LEN>
void gelu_f16(half *x, half *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 1)
        gelu_f16_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
    else if constexpr (VEC_LEN == 8)
        gelu_f16x8_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
    else
        static_assert(VEC_LEN != VEC_LEN, "Invalid VEC_LEN");
}

// -------------------------------------- API --------------------------------------

template <class T, int VEC_LEN>
void gelu(T *x, T *y, int const N)
{
    if constexpr (std::is_same_v<T, float>)
        gelu_f32<VEC_LEN>(x, y, N);
    else if constexpr (std::is_same_v<T, half>)
        gelu_f16<VEC_LEN>(x, y, N);
    else
        static_assert(!std::is_same_v<T, T>, "Unsupported type");
}

template <class T>
void gelu(T *x, T *y, int const N)
{
    for (int i = 0; i < N; ++i)
    {
        T val = x[i];
        T min_val, max_val;

        // 根据数据类型确定截断范围
        if constexpr (std::is_same_v<T, float>)
        {
            min_val = MIN_EXP_F32;
            max_val = MAX_EXP_F32;
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            min_val = MIN_EXP_F16;
            max_val = MAX_EXP_F16;
        }
        else
        {
            static_assert(std::is_same_v<T, T>, "Unsupported type");
        }

        // 截断输入值到[min_val, max_val]
        if (val < min_val)
        {
            val = min_val;
        }
        else if (val > max_val)
        {
            val = max_val;
        }

        // 转换为float进行计算
        float val_float;
        if constexpr (std::is_same_v<T, float>)
        {
            val_float = val;
        }
        else
        {
            val_float = __half2float(val);
        }

        // 计算GELU的tanh近似
        float x_cubed = val_float * val_float * val_float;
        float inner = SQRT_2_PI * (val_float + 0.044715f * x_cubed);
        float tanh_val = tanhf(inner);
        float result = 0.5f * val_float * (1.0f + tanh_val);

        // 转换回原始类型并存储结果
        if constexpr (std::is_same_v<T, float>)
        {
            y[i] = result;
        }
        else
        {
            y[i] = __float2half(result);
        }
    }
}
// -------------------------------------- END --------------------------------------