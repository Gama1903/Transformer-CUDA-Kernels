#include <iostream>
#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>

TEST(execution_config, execution_config_0)
{
    int grid_size;
    for (size_t n = 1024; n <= N; n *= 4)
    {
        std::cout << "n: " << n << "\n";
        {
            CUDA_CHECK(get_grid_size<true>(n, &grid_size, 128));
            std::cout << "grid_size: " << grid_size << "\n";
            std::cout << "\n";
        }
    }
}