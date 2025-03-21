#include <gtest/gtest.h>
#include <kernel_samples/base/common.cuh>
#include <kernel_samples/elementwise/elementwise_add.cuh>

template <class T>
struct ElementwiseAdd : public ::testing::Test
{
protected:
    CudaVector<T> a_, b_, c_, ref;

    ElementwiseAdd() : a_(N, MIN_VALUE, MAX_VALUE), b_(N, MIN_VALUE, MAX_VALUE), c_(N), ref(N) {}

    template <class Func>
    void Verify(Func func)
    {
        CudaVector<T> a(a_), b(b_), c(c_);
        func(a.data(), b.data(), c.data(), N);
        cudaDeviceSynchronize();

        Verifier<T, N> vrf;
        vrf(ref.data(), c.data());
    }

    void SetUp() override
    {
        elementwise_add<T>(a_.data(), b_.data(), ref.data(), N);
    }
};
using ElementwiseAddFloat = ElementwiseAdd<float>;
using ElementwiseAddHalf = ElementwiseAdd<half>;

TEST_F(ElementwiseAddFloat, verify_float)
{
    Verify(elementwise_add<float, 1>);
    Verify(elementwise_add<float, 4>);
}

TEST_F(ElementwiseAddHalf, verify_half)
{
    Verify(elementwise_add<half, 1>);
    Verify(elementwise_add<half, 8>);
}

// // TODO: Profile function
// // TODO: Test for profile