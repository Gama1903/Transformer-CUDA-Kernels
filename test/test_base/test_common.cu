#include <gtest/gtest.h>
#include <base/common.cuh>

TEST(CommonTest, CublasHandleTest)
{
    CublasHandle handle;
}

TEST(CommonTest, CheckCublasTest)
{
    CHECK_CUBLAS(cublasSetStream(NULL, 0)); // FAILED
}

TEST(CommonTest, CudaVectorTest)
{
    CudaVector<float> vec_f(10, 1);
    vec_f.print(5);
    CudaVector<half> vec_h(10, 2);
    vec_h.print(5);
    CudaVector<double> vec_d(10, 3);
    vec_d.print(5);
    CudaVector<int> vec_i(10, 4);
    vec_i.print(5);
}

TEST(CommonTest, CheckCudaTest)
{
    CHECK_CUDA(cudaMalloc(NULL, 0)); // FAILED
}

TEST(CommonTest, getGridSizeTest)
{
    auto func = [&](int &n, int &grid_size)
    {
        n = 256;
        int grid_size_pre;
        CHECK_CUDA(getGridSize(n, &grid_size_pre));
        n <<= 2;
        int grid_size_cur;
        CHECK_CUDA(getGridSize(n, &grid_size_cur));
        n <<= 2;
        while (true)
        {
            int grid_size_next;
            CHECK_CUDA(getGridSize(n, &grid_size_next));
            if (grid_size_next != grid_size_cur)
            {
                grid_size_pre = grid_size_cur;
                grid_size_cur = grid_size_next;
                n <<= 2;
            }
            else
            {
                break;
            }
        }
        grid_size = grid_size_pre;
        n = n >> 4;
    };

    int n, grid_size;
    func(n, grid_size);
    printf("N = %d, grid_size = %d\n", n, grid_size);
}