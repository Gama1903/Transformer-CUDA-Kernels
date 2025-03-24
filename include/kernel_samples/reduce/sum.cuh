#pragma once

#include <cuda_fp16.h>
#include <kernel_samples/base/common.cuh>

// Version cpu
void sum_cpu(float *x, float *y, int const N)
{
    for (int i = 0; i < N; ++i)
        y[0] += x[i];
}

// Version 0: Global memory
__global__ void sum_v0_kernel(float *x, float *y, int const N)
{
    float *x_block = x + blockIdx.x * blockDim.x;
    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        if ((threadIdx.x & (2 * offset - 1)) == 0)
        {
            float val = threadIdx.x + offset < blockDim.x ? x_block[threadIdx.x + offset] : 0.0f;
            x_block[threadIdx.x] += val;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(y, x_block[threadIdx.x]);
}

void sum_v0(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size));
    sum_v0_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
}

// Version 1: Shared memory
__global__ void sum_v1_kernel(float *x, float *y, int const N)
{
    __shared__ float smem[BLOCK_SIZE];
    float *x_block = x + blockIdx.x * blockDim.x;
    smem[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x < N ? x_block[threadIdx.x] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        if ((threadIdx.x & (2 * offset - 1)) == 0)
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(y, smem[threadIdx.x]);
}

void sum_v1(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size));
    sum_v1_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
}

// Version 2: Solve bank conflict
__global__ void sum_v2_kernel(float *x, float *y, int const N)
{
    __shared__ float smem[BLOCK_SIZE];
    float *x_block = x + blockIdx.x * blockDim.x;
    smem[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x < N ? x_block[threadIdx.x] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(y, smem[threadIdx.x]);
}

void sum_v2(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size));
    sum_v2_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
}

// Version 3: Solve idle threads
__global__ void sum_v3_kernel(float *x, float *y, int const N)
{
    __shared__ float smem[BLOCK_SIZE];
    float *x_block = x + blockIdx.x * blockDim.x * 2;
    float val_1 = blockIdx.x * blockDim.x * 2 + threadIdx.x < N ? x_block[threadIdx.x] : 0.0f;
    float val_2 = blockIdx.x * blockDim.x * 2 + threadIdx.x + blockDim.x < N ? x_block[threadIdx.x + blockDim.x] : 0.0f;
    smem[threadIdx.x] = val_1 + val_2;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(y, smem[threadIdx.x]);
}

void sum_v3(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size));
    sum_v3_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
}

// Version 4: Warp shuffle
__forceinline__ __device__ float warp_sum(float val)
{
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}
// register -> shared memory -> register
__global__ void sum_v4_kernel(float *x, float *y, int const N)
{
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int constexpr NUM_WARPS = CEIL_DIV(BLOCK_SIZE, WARP_SIZE);
    __shared__ float smem[NUM_WARPS];

    int warp_idx = tid / WARP_SIZE;
    int lane_idx = tid % WARP_SIZE;

    // register -> shared memory
    float sum = (global_tid < N) ? x[global_tid] : 0.0f;
    sum = warp_sum(sum);
    if (lane_idx == 0)
        smem[warp_idx] = sum;
    __syncthreads();

    // shared memory -> register
    sum = (lane_idx < NUM_WARPS) ? smem[lane_idx] : 0.0f;
    if (warp_idx == 0)
        sum = warp_sum(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

void sum_v4(float *x, float *y, int const N)
{
    int grid_size;
    CUDA_CHECK(get_grid_size(N, &grid_size));
    sum_v4_kernel<<<grid_size, BLOCK_SIZE>>>(x, y, N);
}
