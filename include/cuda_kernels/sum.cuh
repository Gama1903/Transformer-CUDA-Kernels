#pragma once

#include <base/common.cuh>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sum_global_mem_f32_kernel(float *x, float *y, int N)
{
    int block_start_idx = blockIdx.x * blockDim.x;
    float *x_block_start = x + block_start_idx;

    int num_block_elements = min(blockDim.x, N - block_start_idx);

    // 处理奇数个元素的情况
    if (num_block_elements & 1)
    {
        if (threadIdx.x == 0)
        {
            x_block_start[0] += x_block_start[num_block_elements - 1];
        }
        __syncthreads();
        num_block_elements--; // 调整为偶数以便后续归约
    }

    // 核心算法，折半规约
    for (int offset = num_block_elements >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            x_block_start[threadIdx.x] += x_block_start[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(y, x_block_start[0]);
}

void sum_global_mem_f32(float *x, float *y, int N)
{
    int grid_size, block_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, sum_global_mem_f32_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size));

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));
    sum_global_mem_f32_kernel<<<grid_size, block_size>>>(x, y, N);
}

__global__ void sum_shared_mem_f32_kernel(float *x, float *y, int N)
{
    int block_start_idx = blockIdx.x * blockDim.x * 2;
    float *x_block_start = x + block_start_idx;
    int num_block_elements = min(blockDim.x, N - block_start_idx);

    // 加载到共享内存
    extern __shared__ float smem[]; // 动态分配共享内存
    // 加载同时计算，解决线程闲置问题
    float val_1 = (threadIdx.x + block_start_idx < N) ? x_block_start[threadIdx.x] : 0.0f;
    float val_2 = (threadIdx.x + blockDim.x + block_start_idx < N) ? x_block_start[threadIdx.x + blockDim.x] : 0.0f;
    smem[threadIdx.x] = val_1 + val_2;
    __syncthreads();

    if (num_block_elements & 1)
    {
        if (threadIdx.x == 0)
        {
            smem[0] += smem[num_block_elements - 1];
        }
        __syncthreads();
        num_block_elements--;
    }

    for (int offset = num_block_elements >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(y, smem[0]);
}

void sum_shared_mem_f32(float *x, float *y, int N)
{
    int grid_size, block_size, shared_mem_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, sum_shared_mem_f32_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size));

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));

    shared_mem_size = block_size * sizeof(float);
    sum_shared_mem_f32_kernel<<<grid_size, block_size, shared_mem_size>>>(x, y, N);
}

__forceinline__ __device__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void sum_block_all_f32_kernel(float *x, float *y, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int num_warps = CDIV(blockDim.x, WARP_SIZE);
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int warp_idx = tid / WARP_SIZE;
    int lane_idx = tid % WARP_SIZE;

    float sum = (idx < N) ? x[idx] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_idx == 0)
        smem[warp_idx] = sum;
    __syncthreads();

    sum = (lane_idx < num_warps) ? smem[lane_idx] : 0.0f;
    if (warp_idx == 0)
        sum = warp_reduce_sum(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

void sum_block_all_f32(float *x, float *y, int N)
{
    int grid_size, block_size, shared_mem_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, sum_block_all_f32_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size));

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));

    shared_mem_size = block_size * sizeof(float);
    sum_block_all_f32_kernel<<<grid_size, block_size, shared_mem_size>>>(x, y, N);
}

__global__ void sum_block_all_f32x4_kernel(float *x, float *y, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int vec_start_idx = idx * 4;
    int num_warps = CDIV(blockDim.x, WARP_SIZE);
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int warp_idx = tid / WARP_SIZE;
    int lane_idx = tid % WARP_SIZE;

    // 向量化加载
    float sum = 0.0f;
    if (vec_start_idx + 3 < N)
    {
        float4 x_vec = FLOAT4(x[vec_start_idx]);
        sum = x_vec.x + x_vec.y + x_vec.z + x_vec.w;
    }
#pragma once
    for (int i = 0; i < 3; i++)
        if (vec_start_idx + i < N)
            sum = x[vec_start_idx + i];

    sum = warp_reduce_sum(sum);
    if (lane_idx == 0)
        smem[warp_idx] = sum;
    __syncthreads();

    sum = (lane_idx < num_warps) ? smem[lane_idx] : 0.0f;
    if (warp_idx == 0)
        sum = warp_reduce_sum(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

void sum_block_all_f32x4(float *x, float *y, int N)
{
    int grid_size, block_size, shared_mem_size;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, sum_block_all_f32x4_kernel));
    block_size = 1 << (31 - __builtin_clz(block_size));
    block_size /= 4;

    CHECK_CUDA(getGridSize(N, &grid_size, block_size));

    shared_mem_size = block_size * sizeof(float);
    sum_block_all_f32_kernel<<<grid_size, block_size, shared_mem_size>>>(x, y, N);
}