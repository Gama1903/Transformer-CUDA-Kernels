#pragma once

#include <iostream>

#define BLOCK_SIZE 128
#define WAVE_NUM 32

constexpr int ceil_div(int n, int d)
{
    return (n + d - 1) / d;
}

// wave_num = 32
// block_size = 128, 256, 512
template <bool DEBUG = false>
__forceinline__ cudaError_t get_grid_size(size_t n, int *grid_size, int block_size = BLOCK_SIZE, int wave_num = WAVE_NUM)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_num;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int thread_per_sm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&thread_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if constexpr (DEBUG)
    {
        std::cout << "dev: " << dev << "\n";
        std::cout << "sm_num: " << sm_num << "\n";
        std::cout << "thread_per_sm: " << thread_per_sm << "\n";
        std::cout << "grid_size_small: " << ceil_div(n, block_size) << "\n";
        std::cout << "grid_size_large: " << sm_num * thread_per_sm / block_size * wave_num << "\n";
    }

    *grid_size = std::max(1, std::min(
                                 ceil_div(n, block_size),
                                 sm_num * thread_per_sm / block_size * wave_num));
    return cudaSuccess;
}