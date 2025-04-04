#include <gtest/gtest.h>
#include <base/common.cuh>
#include <cuda_kernels/sum.cuh>

// 数值检查次数
constexpr int num_runs = 25;
// 最大数据长度
#ifdef ALIGNED_DATA
constexpr int N = 1024 * 1024;
#else
constexpr int N = 1e6 + 1;
#endif

void sum_cpu(float *x, float *y, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += x[i];
    *y = sum;
}

void sum_cublas(float *x, float *y, int N, cublasHandle_t handle, float *ones)
{
    CHECK_CUBLAS(cublasSdot(handle, N, x, 1, ones, 1, y));
}

struct SumVerify : public ::testing::Test
{
    CudaVector<float> x, y;
    CudaVector<float> ref;
    SumVerify() : ref(1, N) {};
    void SetUp() override
    {
        printf("N: %d\n", N);
    }
};

TEST_F(SumVerify, Cpu)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_cpu(x.data(), y.data(), N);
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

TEST_F(SumVerify, Cublas)
{
    CublasHandle handle;
    CudaVector<float> ones(N, 1.0f);

    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_cublas(x.data(), y.data(), N, handle.get(), ones.data());
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

TEST_F(SumVerify, globalMemF32)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_global_mem_f32(x.data(), y.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

TEST_F(SumVerify, SharedMemF32)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_shared_mem_f32(x.data(), y.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

TEST_F(SumVerify, BlockAllF32)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_block_all_f32(x.data(), y.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

TEST_F(SumVerify, BlockAllF32x4)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };

    for (int runs = 0; runs < num_runs; ++runs)
    {
        setup();
        sum_block_all_f32x4(x.data(), y.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(y.data(), ref.data(), 1, 1e-5f);
    }
}

static float lower_perf = 0.0f;
static float upper_perf = 0.0f;

struct SumProfile : public ::testing::Test
{
    CudaVector<float> x, y;
};

TEST_F(SumProfile, Cpu)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };
    auto sum = [&]()
    {
        sum_cpu(x.data(), y.data(), N);
    };

    CudaBenchmark benchmark;
    auto result = benchmark.run(setup, sum);
    lower_perf = result.avg_time;
    printf("Lower performance(cpu):\n");
    benchmark.print_result(result);
}

TEST_F(SumProfile, Cublas)
{
    CublasHandle handle;
    CudaVector<float> ones(N, 1.0f);

    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };
    auto sum = [&]()
    {
        sum_cublas(x.data(), y.data(), N, handle.get(), ones.data());
    };

    CudaBenchmark benchmark;
    auto result_cub = benchmark.run(setup, sum);
    upper_perf = result_cub.avg_time;
    printf("Upper performance(cublas):\n");
    benchmark.print_result(result_cub);
}

TEST_F(SumProfile, SharedMemF32)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };
    auto sum_shared_mem = [&]()
    {
        sum_shared_mem_f32(x.data(), y.data(), N);
    };

    CudaBenchmark benchmark;
    auto result_shared_mem = benchmark.run(setup, sum_shared_mem);
    printf("Shared memory result:\n");
    benchmark.print_result(result_shared_mem);
    printf("Speedup(lower): %.2fx\nSpeedup(upper): %.2fx\n", lower_perf / result_shared_mem.avg_time, upper_perf / result_shared_mem.avg_time);
}

TEST_F(SumProfile, BlockAllF32)
{
    auto setup = [&]()
    {
        x = CudaVector<float>(N, 1);
        y = CudaVector<float>(1, 0);
    };
    auto sum_warp_shfl = [&]()
    {
        sum_block_all_f32(x.data(), y.data(), N);
    };

    CudaBenchmark benchmark;
    auto result_warp_shfl = benchmark.run(setup, sum_warp_shfl);
    printf("Warp shuffle result:\n");
    benchmark.print_result(result_warp_shfl);
    printf("Speedup(lower): %.2fx\nSpeedup(upper): %.2fx\n", lower_perf / result_warp_shfl.avg_time, upper_perf / result_warp_shfl.avg_time);
}
