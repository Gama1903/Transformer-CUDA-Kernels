#pragma once

#include <cuda_fp16.h>
#include <base/common.cuh>

#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])

__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x)
        c[idx] = a[idx] + b[idx];
}

void elementwise_add_f32(float *a, float *b, float *c, int N)
{
    int grid_size;
    CHECK_CUDA(getGridSize(N, &grid_size));
    elementwise_add_f32_kernel<<<grid_size, block_size>>>(a, b, c, N);
}

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N)
{
    int num_float4 = (N + 3) / 4;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_float4; idx += blockDim.x * gridDim.x)
    {
        int vec_idx = idx * 4;
        if (vec_idx + 3 < N)
        {
            float4 a_vec = FLOAT4(a[vec_idx]);
            float4 b_vec = FLOAT4(b[vec_idx]);

            float4 c_vec;
            c_vec.x = a_vec.x + b_vec.x;
            c_vec.y = a_vec.y + b_vec.y;
            c_vec.z = a_vec.z + b_vec.z;
            c_vec.w = a_vec.w + b_vec.w;

            FLOAT4(c[vec_idx]) = c_vec;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
                if (vec_idx + i < N)
                    c[vec_idx + i] = a[vec_idx + i] + b[vec_idx + i];
        }
    }
}

void elementwise_add_f32x4(float *a, float *b, float *c, int N)
{
    int num_float4 = (N + 3) / 4;
    int grid_size;
    CHECK_CUDA(getGridSize(num_float4, &grid_size));
    elementwise_add_f32x4_kernel<<<grid_size, block_size>>>(a, b, c, N);
}