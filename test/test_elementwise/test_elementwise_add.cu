#include <gtest/gtest.h>
#include <kernel_samples/base/general.cuh>
#include <kernel_samples/elementwise/elementwise_add.cuh>

#define N N_

template <class T, size_t N>
void elementwise_add_cpu(T const *__restrict__ a, T const *__restrict__ b, T *__restrict__ c)
{
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
}

template <class T>
class ElementwiseAdd : public ::testing::Test
{
protected:
    VectorSet<T> vs;
    Verifier<T> vrf;
    Profiler prfl;

    float dur_baseline;
    int num_check;

    ElementwiseAdd() : vs{N}, vrf{}, prfl{}, dur_baseline(0), num_check(2) {}

    void SetUp() override
    {
        dur_baseline = prfl.duration_cpu(elementwise_add_cpu<T, N>, vs.a.data(), vs.b.data(), vs.c_ref.data());
        std::cout << "elementwise_add_cpu duration: " << dur_baseline << " us" << "\n";
    }

    void TearDown() override
    {
    }
};

class ElementwiseAddF32 : public ElementwiseAdd<float>
{
};

TEST_F(ElementwiseAddF32, elementwise_add_f32)
{
    float dur_elementwise_add_f32 = prfl.duration_gpu(elementwise_add_f32<N, 1>, vs.a.data(), vs.b.data(), vs.c.data());
    std::cout << "elementwise_add_f32_kernel duration: " << dur_elementwise_add_f32 << " us" << "\n";
    vrf.check(vs.c_ref.data(), vs.c.data(), num_check);
    std::cout << "Speedup: " << dur_baseline / dur_elementwise_add_f32 << "x\n";
    vs.reset();

    float dur_elementwise_add_f32x4 = prfl.duration_gpu(elementwise_add_f32<N, 4>, vs.a.data(), vs.b.data(), vs.c.data());
    std::cout << "elementwise_add_f32x4_kernel duration: " << dur_elementwise_add_f32x4 << " us" << "\n";
    vrf.check(vs.c_ref.data(), vs.c.data(), num_check);
    std::cout << "Speedup: " << dur_baseline / dur_elementwise_add_f32x4 << "x\n";
}

class ElementwiseAddF16 : public ElementwiseAdd<half>
{
};

TEST_F(ElementwiseAddF16, elementwise_add_f16)
{
    float dur_elementwise_add_f16 = prfl.duration_gpu(elementwise_add_f16<N, 1>, vs.a.data(), vs.b.data(), vs.c.data());
    std::cout << "elementwise_add_f16_kernel duration: " << dur_elementwise_add_f16 << " us" << "\n";
    vrf.check(vs.c_ref.data(), vs.c.data(), num_check);
    std::cout << "Speedup: " << dur_baseline / dur_elementwise_add_f16 << "x\n";
    vs.reset();

    float dur_elementwise_add_f16x8 = prfl.duration_gpu(elementwise_add_f16<N, 8>, vs.a.data(), vs.b.data(), vs.c.data());
    std::cout << "elementwise_add_f16x8_kernel duration: " << dur_elementwise_add_f16x8 << " us" << "\n";
    vrf.check(vs.c_ref.data(), vs.c.data(), num_check);
    std::cout << "Speedup: " << dur_baseline / dur_elementwise_add_f16x8 << "x\n";
}