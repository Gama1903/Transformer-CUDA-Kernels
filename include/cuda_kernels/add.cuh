#pragma once

#include <base/common.cuh>

#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])

__global__ void add_f32_kernel(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

void add_f32(float *a, float *b, float *c, int N)
{
    int grid_size, block_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, add_f32_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size)); // 将block_size设置为2的幂次, 且不大于当前值

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));
    add_f32_kernel<<<grid_size, block_size>>>(a, b, c, N);
}

__global__ void add_f32x4_kernel(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_start_idx = idx * 4; // 向量化元素起始索引
    // 处理向量化对齐的元素
    if (vec_start_idx + 3 < N)
    {
        float4 a_vec = FLOAT4(a[vec_start_idx]);
        float4 b_vec = FLOAT4(b[vec_start_idx]);

        float4 c_vec;
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;

        FLOAT4(c[vec_start_idx]) = c_vec;
    }
    // 处理尾端未对齐的元素
    if (vec_start_idx < N)
        c[vec_start_idx] = a[vec_start_idx] + b[vec_start_idx];
    if (vec_start_idx + 1 < N)
        c[vec_start_idx + 1] = a[vec_start_idx + 1] + b[vec_start_idx + 1];
    if (vec_start_idx + 2 < N)
        c[vec_start_idx + 2] = a[vec_start_idx + 2] + b[vec_start_idx + 2];
}

void add_f32x4(float *a, float *b, float *c, int N)
{
    int grid_size, block_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, add_f32x4_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size));
    block_size /= 4; // 每个线程处理4个元素，线程块尺寸相应缩小

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));
    add_f32x4_kernel<<<grid_size, block_size>>>(a, b, c, N);
}