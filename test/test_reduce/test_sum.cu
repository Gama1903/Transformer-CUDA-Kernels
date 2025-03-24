#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>
#include <kernel_samples/reduce/sum.cuh>

struct Sum : public ::testing::Test
{
protected:
    CudaVector<float> x_;
    CudaVector<float> y_, ref;

    Sum() : x_(N, MIN_VALUE, MAX_VALUE), y_(1), ref(1) {}

    template <class Func>
    void Verify(Func func)
    {
        CudaVector<float> x(x_);
        CudaVector<float> y(y_);
        func(x.data(), y.data(), N);
        cudaDeviceSynchronize();

        Verifier<float, 1> vrf;
        vrf(ref.data(), y.data(), 1.0f);

        std::cout << "Res: " << "\n";
        y.print(1);
    }

    void SetUp() override
    {
        CudaVector<float> x = x_.to_float();
        sum_cpu(x.data(), ref.data(), N);

        std::cout << "Ref: " << "\n";
        ref.print(1);
    }
};

TEST_F(Sum, verify)
{
    Verify(sum_v0);
    Verify(sum_v1);
    Verify(sum_v2);
    Verify(sum_v3);
    Verify(sum_v4);
}