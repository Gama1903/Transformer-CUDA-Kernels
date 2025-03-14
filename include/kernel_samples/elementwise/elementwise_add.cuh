#pragma once

#include <cuda_fp16.h>
#include <kernel_samples/base/general.cuh>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32 --------------------------------------

template <int N>
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c)
{
    int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx < N)
        c[global_idx] = a[global_idx] + b[global_idx];
}

// N / 4 == 1
template <int N>
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c)
{
    int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int const vec_base_idx = global_idx * 4;

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
template <int N, int VEC_LEN>
void elementwise_add_f32(float *a, float *b, float *c)
{
    int grid_size;
    CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 4)
        elementwise_add_f32x4_kernel<N><<<grid_size, BLOCK_SIZE>>>(a, b, c);
    else if constexpr (VEC_LEN == 1)
        elementwise_add_f32_kernel<N><<<grid_size, BLOCK_SIZE>>>(a, b, c);
}

// -------------------------------------- FP16 --------------------------------------

template <int N>
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c)
{
    int const global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_idx < N)
        c[global_idx] = __hadd(a[global_idx], b[global_idx]);
}

// N / 8 == 1
template <int N>
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c)
{
    int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int const vec_base_idx = global_idx * 8;
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
template <int N, int VEC_LEN>
void elementwise_add_f16(half *a, half *b, half *c)
{
    int grid_size;
    CHECK(get_grid_size(N, &grid_size, BLOCK_SIZE * VEC_LEN));
    if constexpr (VEC_LEN == 8)
        elementwise_add_f16x8_kernel<N><<<grid_size, BLOCK_SIZE>>>(a, b, c);
    else if constexpr (VEC_LEN == 1)
        elementwise_add_f16_kernel<N><<<grid_size, BLOCK_SIZE>>>(a, b, c);
}