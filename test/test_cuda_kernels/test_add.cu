#include <gtest/gtest.h>
#include <base/common.cuh>
#include <cuda_kernels/add.cuh>

// 数值检查次数
constexpr int num_runs = 25;
// 最大数据长度
#ifdef ALIGNED_DATA
constexpr int N = 1024 * 1024;
#else
constexpr int N = 1e6 + 1;
#endif

void add_cpu(float *a, float *b, float *c, int N)
{
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

void add_cublas(float *a, float *b, float *c, int N, cublasHandle_t handle, float alpha, float beta)
{
    CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, &alpha, a, N, &beta, b, N, c, N));
}

struct AddVerify : public ::testing::Test
{
    CudaVector<float> a, b, c, ref;
    AddVerify() : ref(N, 3.0f) {}
    void SetUp() override
    {
        printf("N: %d\n", N);
    }
};

TEST_F(AddVerify, Cpu)
{
    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };

    for (int runs = 0; runs < num_runs; runs++)
    {
        setup();
        add_cpu(a.data(), b.data(), c.data(), N);
        verify(c.data(), ref.data(), N, 1e-5f);
    }
}

TEST_F(AddVerify, Cublas)
{
    CublasHandle handle;

    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };

    for (int runs = 0; runs < num_runs; runs++)
    {
        setup();
        add_cublas(a.data(), b.data(), c.data(), N, handle.get(), 1.0f, 1.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(c.data(), ref.data(), N, 1e-5f);
    }
}

TEST_F(AddVerify, AddF32)
{
    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };

    for (int runs = 0; runs < num_runs; runs++)
    {
        setup();
        add_f32(a.data(), b.data(), c.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(c.data(), ref.data(), N, 1e-5f);
    }
}

TEST_F(AddVerify, AddF32x4)
{
    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };

    for (int runs = 0; runs < num_runs; runs++)
    {
        setup();
        add_f32x4(a.data(), b.data(), c.data(), N);
        CHECK_CUDA(cudaDeviceSynchronize());
        verify(c.data(), ref.data(), N, 1e-5f);
    }
}

static float upper_perf = 0;
static float lower_perf = 0;

struct AddProfile : public ::testing::Test
{
    CudaVector<float> a, b, c;
    void SetUp() override
    {
        printf("N: %d\n", N);
    }
};

TEST_F(AddProfile, Cpu)
{
    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };
    auto add = [&]()
    {
        add_cpu(a.data(), b.data(), c.data(), N);
    };

    CudaBenchmark benchmark;
    auto result = benchmark.run(setup, add);
    lower_perf = result.avg_time;
    printf("Lower performance(cpu):\n");
    benchmark.print_result(result);
}

TEST_F(AddProfile, Cublas)
{
    CublasHandle handle;

    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };
    auto add = [&]()
    {
        add_cublas(a.data(), b.data(), c.data(), N, handle.get(), 1.0f, 1.0f);
    };

    CudaBenchmark benchmark;
    auto result = benchmark.run(setup, add);
    upper_perf = result.avg_time;
    printf("Upper performance(cublas):\n");
    benchmark.print_result(result);
}

TEST_F(AddProfile, AddF32)
{
    auto setup = [&]()
    {
        a = CudaVector<float>(N, 1);
        b = CudaVector<float>(N, 2);
        c = CudaVector<float>(N, 0);
    };
    auto add = [&]()
    {
        add_f32(a.data(), b.data(), c.data(), N);
    };

    CudaBenchmark benchmark;
    auto result = benchmark.run(setup, add);
    printf("Add f32 performance:\n");
    benchmark.print_result(result);
    printf("Speedup(lower): %.2fx\nSpeedup(upper): %.2fx\n", lower_perf / result.avg_time, upper_perf / result.avg_time);
}