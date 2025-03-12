#pragma once

#include <cuda_fp16.h>                              // half
#include <iostream>                                 // cout
#include <chrono>                                   // cpu 计时
#include <functional>                               // 函数式方法
#include <vector>                                   // 向量内存管理
#include <kernel_samples/base/error.cuh>            // 自定义 CUDA 错误处理
#include <kernel_samples/base/execution_config.cuh> // 自定义执行配置设置
#include <gtest/gtest.h>                            // EXPECT_NEAR 浮点数比较

#define N_ MAX_ELEMENT_NUM

// 定义最大元素数量，防止显存溢出
constexpr size_t MAX_ELEMENT_NUM = 1024 * 1024;

// 类型萃取
template <class T>
struct TypeTraits;

template <>
struct TypeTraits<float>
{
    static constexpr float epsilon = 1e-3f; // 误差范围，用于浮点数比较

    static float error(float const x, float const y)
    {
        return std::abs(x - y); // float 类型使用 STL 库函数
    }

    static float to_printable(float const x)
    {
        return x;
    }
};

template <>
struct TypeTraits<half>
{
    static constexpr float epsilon = 1e-1f; // 半精度浮点放宽误差范围

    static float error(half const x, half const y)
    {
        return __half2float(__habs(x - y)); // half 类型使用 CUDA 库函数，并转换为 float 类型进行比较
    }

    static float to_printable(half const x)
    {
        return __half2float(x); // STL 库输出流不支持 half 类型，需要转换为 float 类型
    }
};

// 结果验证器
template <class T>
struct Verifier
{
    // 在误差范围内检验结果，n <= N / 2
    void check(T const *ref, T const *res, size_t n) const
    {
        for (size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(ref[i], res[i], TypeTraits<T>::epsilon); // 使用 GTest 框架进行浮点数比较
        }
        for (size_t i = N_ - n; i < N_; i++)
        {
            EXPECT_NEAR(ref[i], res[i], TypeTraits<T>::epsilon);
        }
    }

    // 打印尾部结果
    void print(T const *res, size_t N, size_t n = 128) const
    {
        for (size_t i = N - n; i < N; i++)
        {
            std::cout << TypeTraits<T>::to_printable(res[i]) << " ";
            if ((i + 1) % 64 == 0)
                std::cout << "\n";
        }
        std::cout << "\n";
    }
};

// TODO 性能分析器的实现逻辑仍有缺陷
// 性能分析器
struct Profiler
{
    size_t const num_warmup_;
    size_t const num_test_;

    explicit Profiler(size_t num_test = 40, size_t num_warmup = 3)
        : num_test_(num_test), num_warmup_(num_warmup) {}

    template <class Func, class... Args>
    float duration_cpu(Func const &func, Args &&...args)
    {
        for (size_t i = 0; i < num_warmup_; ++i)
        {
            std::invoke(func, std::forward<Args>(args)...);
        }

        size_t num_test_cpu = num_test_ / 4;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_test_cpu; ++i)
        {
            std::invoke(func, std::forward<Args>(args)...);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count() / num_test_cpu;
    }

    template <class Func, class... Args>
    float duration_gpu(Func const &func, Args &&...args)
    {
        for (size_t i = 0; i < num_warmup_; ++i)
        {
            std::invoke(func, std::forward<Args>(args)...);
        }
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start, 0));
        for (size_t i = 0; i < num_test_; ++i)
        {
            std::invoke(func, std::forward<Args>(args)...);
        }
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));

        float duration;
        CHECK(cudaEventElapsedTime(&duration, start, stop));
        return duration / num_test_ * 1000.0f;
    }
};

// 自定义分配器，使用 CUDA 统一内存管理
template <class T>
struct CudaAllocator
{
    using value_type = T;

    T *allocate(size_t size)
    {
        T *ptr = nullptr;
        CHECK(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0)
    {
        CHECK(cudaFree(ptr));
    }

    // 显式确定初始化方式，避免 CPU 上低效的零初始化
    template <class... Args>
    void construct(T *p, Args &&...args)
    {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
        {
            ::new ((void *)p) T(std::forward<Args>(args)...);
        }
    }
};

template <class T>
struct VectorSet
{
    // 向量内存管理，遵循 RAII 机制
    std::vector<T, CudaAllocator<T>> a;
    std::vector<T, CudaAllocator<T>> b;
    std::vector<T, CudaAllocator<T>> c;
    std::vector<T, CudaAllocator<T>> c_ref;

    explicit VectorSet(size_t n)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            a.resize(n, 1.0f);
            b.resize(n, 2.0f);
            c.resize(n, 0.0f);
            c_ref.resize(n, 0.0f);
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            a.resize(n, __float2half(1.0f));
            b.resize(n, __float2half(2.0f));
            c.resize(n, __float2half(0.0f));
            c_ref.resize(n, __float2half(0.0f));
        }
    }

    void reset()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            std::fill(c.begin(), c.end(), 0.0f);
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            std::fill(c.begin(), c.end(), __float2half(0.0f));
        }
    }
};