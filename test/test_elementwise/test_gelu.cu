#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>
#include <kernel_samples/elementwise/gelu.cuh>

template <class T>
struct Gelu : public ::testing::Test
{
protected:
    CudaVector<T> x_, y_, ref;

    Gelu() : x_(N, MIN_VALUE, MAX_VALUE), y_(N), ref(N) {}

    template <class Func>
    void Verify(Func func)
    {
        CudaVector<T> x(x_), y(y_);
        func(x.data(), y.data(), N);
        cudaDeviceSynchronize();

        Verifier<T, N> vrf;
        vrf(ref.data(), y.data());
    }

    void SetUp() override
    {
        gelu<T>(x_.data(), ref.data(), N);
    }
};

using GeluFloat = Gelu<float>;
using GeluHalf = Gelu<half>;

TEST_F(GeluFloat, verify_float)
{
    Verify(gelu<float, 1>);
    Verify(gelu<float, 4>);
}

TEST_F(GeluHalf, verify_half)
{
    Verify(gelu<half, 1>);
    Verify(gelu<half, 8>);
}