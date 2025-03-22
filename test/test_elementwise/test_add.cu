#include <algorithm>
#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>
#include <kernel_samples/elementwise/add.cuh>

template <class Tp>
struct Add : public ::testing::Test
{
protected:
    CudaVector<Tp> a_, b_, c_;
    CudaVector<float> ref;

    Add() : a_(N, MIN_VALUE, MAX_VALUE), b_(N, MIN_VALUE, MAX_VALUE), c_(N), ref(N) {}

    template <class Func>
    void Verify(Func func)
    {
        CudaVector<Tp> a(a_), b(b_), c(c_);
        func(a.data(), b.data(), c.data(), N);
        cudaDeviceSynchronize();

        Verifier<Tp, N> vrf;
        vrf(ref.data(), c.data());

        std::cout << "Res: " << "\n";
        c.print(8);
    }

    void SetUp() override
    {
        CudaVector<float> a = a_.to_float();
        CudaVector<float> b = b_.to_float();
        add<float, 0>(a.data(), b.data(), ref.data(), N);

        std::cout << "Ref: " << "\n";
        ref.print(8);
    }
};

using AddFloat = Add<float>;
using AddHalf = Add<half>;

TEST_F(AddFloat, verify_float)
{
    Verify(add<float, 1>);
    Verify(add<float, 4>);
}

TEST_F(AddHalf, verify_half)
{
    Verify(add<half, 1>);
    Verify(add<half, 8>);
}

// TODO: Profile function
// TODO: Test for profile