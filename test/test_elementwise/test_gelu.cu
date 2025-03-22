#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>
#include <kernel_samples/elementwise/gelu.cuh>

template <class Tp>
struct Gelu : public ::testing::Test
{
protected:
    CudaVector<Tp> x_, y_;
    CudaVector<float> ref;

    Gelu() : x_(N, MIN_VALUE, MAX_VALUE), y_(N), ref(N) {}

    template <class Func>
    void Verify(Func func)
    {
        CudaVector<Tp> x(x_), y(y_);
        func(x.data(), y.data(), N);
        cudaDeviceSynchronize();

        Verifier<Tp, N> vrf;
        vrf(ref.data(), y.data());

        std::cout << "Res: " << "\n";
        y.print(8);
    }

    void SetUp() override
    {
        CudaVector<float> x = x_.to_float();
        gelu<float, 0>(x.data(), ref.data(), N);

        std::cout << "Ref: " << "\n";
        ref.print(8);
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