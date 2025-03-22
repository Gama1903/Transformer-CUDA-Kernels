#pragma once

#include <random>        // 随机数生成
#include <cuda_fp16.h>   // half
#include <iostream>      // cout
#include <chrono>        // cpu 计时
#include <functional>    // 函数式方法
#include <vector>        // 向量内存管理
#include <gtest/gtest.h> // EXPECT_NEAR 浮点数比较

// 执行配置基本设置
#define BLOCK_SIZE 128
#define WAVE_NUM 32
#define WARP_SIZE 32

// 检查CUDA错误
#define CUDA_CHECK(call)                                                             \
    do                                                                               \
    {                                                                                \
        cudaError_t const error_code = call;                                         \
        if (error_code != cudaSuccess)                                               \
        {                                                                            \
            std::cerr << "CUDA error:\n";                                            \
            std::cerr << "  File:       " << __FILE__ << "\n";                       \
            std::cerr << "  Line:       " << __LINE__ << "\n";                       \
            std::cerr << "  Error code: " << error_code << "\n";                     \
            std::cerr << "  Error text: " << cudaGetErrorString(error_code) << "\n"; \
        }                                                                            \
    } while (0)

// 定义最大元素数量
constexpr int MAX_ELEMENT_NUM = 1024 * 1024;
constexpr int N = MAX_ELEMENT_NUM;
constexpr float MAX_VALUE = 3.0f;
constexpr float MIN_VALUE = -3.0f;

// 计算向上取整的除法
constexpr int ceil_div(int n, int d)
{
    return (n + d - 1) / d;
}

// wave_num = 32
// block_size = 128, 256, 512
template <bool DEBUG = false>
__forceinline__ cudaError_t get_grid_size(int n, int *grid_size, int block_size = BLOCK_SIZE, int wave_num = WAVE_NUM)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_num;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int thread_per_sm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&thread_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    if constexpr (DEBUG)
    {
        std::cout << "dev: " << dev << "\n";
        std::cout << "sm_num: " << sm_num << "\n";
        std::cout << "thread_per_sm: " << thread_per_sm << "\n";
        std::cout << "grid_size_small: " << ceil_div(n, block_size) << "\n";
        std::cout << "grid_size_large: " << sm_num * thread_per_sm / block_size * wave_num << "\n";
    }

    *grid_size = std::max(1, std::min(
                                 ceil_div(n, block_size),
                                 sm_num * thread_per_sm / block_size * wave_num));
    return cudaSuccess;
}

// 类型萃取
template <class Tp>
struct TypeTraits
{
    static_assert(!std::is_same_v<Tp, Tp>, "TypeTraits: Unsupported type!");
};

template <>
struct TypeTraits<float>
{
    static constexpr float epsilon = 1e-5f;
    static constexpr auto from_float = [](float x)
    { return x; };
    static constexpr auto to_float = [](float x)
    { return x; };
};

template <>
struct TypeTraits<half>
{
    static constexpr float epsilon = 1e-1f;
    static constexpr auto from_float = __float2half;
    static constexpr auto to_float = __half2float;
};

template <>
struct TypeTraits<int>
{
    static constexpr auto from_float = [](float x)
    { return static_cast<int>(x); };
    static constexpr auto to_float = [](int x)
    { return static_cast<float>(x); };
};

// 结果验证器
template <class Tp, int N>
struct Verifier
{
    using type_traits = TypeTraits<Tp>;

    void operator()(float const *ref, Tp const *res) const
    {
        float val_res;
        for (int i = 0; i < N; i++)
        {
            val_res = type_traits::to_float(res[i]);
            EXPECT_NEAR(ref[i], val_res, type_traits::epsilon); // 使用 GTest 框架进行浮点数比较
        }
    }

    void operator()(float const *ref, Tp const *res, int n) const
    {
        float val_res;
        for (int i = 0; i < n; i++)
        {
            val_res = type_traits::to_float(res[i]);
            EXPECT_NEAR(ref[i], val_res, type_traits::epsilon);
        }
        for (int i = N - n; i < N; i++)
        {
            val_res = type_traits::to_float(res[i]);
            EXPECT_NEAR(ref[i], val_res, type_traits::epsilon);
        }
    }
};

// 自定义分配器，使用 CUDA 统一内存管理
template <class Tp>
struct CudaAllocator
{
    using value_type = Tp;

    Tp *allocate(int size)
    {
        Tp *ptr = nullptr;
        CUDA_CHECK(cudaMallocManaged(&ptr, size * sizeof(Tp)));
        return ptr;
    }

    void deallocate(Tp *ptr, int size = 0)
    {
        CUDA_CHECK(cudaFree(ptr));
    }
};

// 性能分析标签
struct prof_tag
{
};
// 非性能分析标签
struct no_prof_tag
{
};

// 自定义向量类，用于存储和操作数据
template <class Tp, class Tag = no_prof_tag>
struct CudaVector
{
    using type_traits = TypeTraits<Tp>;

    std::vector<Tp, CudaAllocator<Tp>> vec;

    explicit CudaVector(int size = 0, float value = 0.0f) : vec(size, type_traits::from_float(value)) {}

    explicit CudaVector(std::initializer_list<float> init)
    {
        vec.reserve(init.size());
        for (auto &v : init)
            vec.push_back(type_traits::from_float(v));
    }

    explicit CudaVector(int size, float min_val, float max_val, int seed = 42) : vec(size)
    {
        std::mt19937 engine(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (auto &elem : vec)
        {
            elem = type_traits::from_float(dist(engine));
        }
    }

    CudaVector(CudaVector const &other) noexcept : vec(other.vec) {}
    CudaVector &operator=(CudaVector const &other) noexcept
    {
        vec = other.vec;
        return *this;
    }

    Tp *data() noexcept { return vec.data(); }
    Tp const *data() const noexcept { return vec.data(); }
    int size() const noexcept { return vec.size(); }
    Tp &operator[](int i) noexcept { return vec[i]; }
    Tp const &operator[](int i) const noexcept { return vec[i]; }
    Tp &at(int i) noexcept { return vec.at(i); }
    Tp const &at(int i) const noexcept { return vec.at(i); }
    auto begin() noexcept { return vec.begin(); }
    auto end() noexcept { return vec.end(); }
    bool empty() const noexcept { return vec.empty(); }
    auto front() noexcept { return vec.front(); }
    void push_back(Tp const &value) noexcept { return vec.push_back(value); }

    void reset(float value = 0.0f)
    {
        if (!vec.empty())
        {
            std::fill(vec.begin(), vec.end(), type_traits::from_float(value));
        }
    }

    CudaVector<float> to_float() const
    {
        CudaVector<float> res(vec.size());
        std::transform(vec.begin(), vec.end(), res.begin(), type_traits::to_float);
        return res;
    }

    void to_float(CudaVector<float> &res) const
    {
        std::transform(vec.begin(), vec.end(), res.begin(), type_traits::to_float);
    }

    // void print(Tag tag = Tag{}) const
    // {
    //     if constexpr (std::is_same_v<Tag, prof_tag>)
    //     {
    //         for (int i = 0; i < vec.size(); i += 2)
    //         {
    //             std::cout << "Kernel " << i / 2 + 1 << "\n";
    //             std::cout << "Duration: " << type_traits::to_float(vec[i]) << " us, ";
    //             std::cout << "speedup: " << type_traits::to_float(vec[i + 1]) << "x\n\n";
    //         }
    //     }
    //     else
    //     {
    //         for (auto v : vec)
    //         {
    //             std::cout << type_traits::to_float(v) << " ";
    //         }
    //         std::cout << "\n";
    //     }
    // }

    void print(int n) const
    {
        int size = vec.size();
        for (int i = 0; i < n; i++)
        {
            std::cout << type_traits::to_float(vec[i]) << " ";
        }
        std::cout << "\n";
        for (int i = size - n; i < size; i++)
        {
            std::cout << type_traits::to_float(vec[i]) << " ";
        }
        std::cout << "\n";
    }
};

// // TODO 逻辑有缺陷
// // 性能分析器
// struct Profiler
// {
//     int const num_warmup_;
//     int const num_test_;

//     explicit Profiler(int num_test = 40, int num_warmup = 3)
//         : num_test_(num_test), num_warmup_(num_warmup) {}

//     template <class Func, class... Args>
//     float duration_cpu(Func const &func, Args &&...args)
//     {
//         for (int i = 0; i < num_warmup_; ++i)
//         {
//             std::invoke(func, std::forward<Args>(args)...);
//         }

//         int num_test_cpu = num_test_ / 4;
//         auto cpu_start = std::chrono::high_resolution_clock::now();
//         for (int i = 0; i < num_test_cpu; ++i)
//         {
//             std::invoke(func, std::forward<Args>(args)...);
//         }
//         auto cpu_end = std::chrono::high_resolution_clock::now();

//         return std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count() / num_test_cpu;
//     }

//     template <class Func, class... Args>
//     float duration_gpu(Func const &func, Args &&...args)
//     {
//         for (int i = 0; i < num_warmup_; ++i)
//         {
//             std::invoke(func, std::forward<Args>(args)...);
//         }
//         CHECK(cudaDeviceSynchronize());

//         cudaEvent_t start, stop;
//         CHECK(cudaEventCreate(&start));
//         CHECK(cudaEventCreate(&stop));

//         CHECK(cudaEventRecord(start, 0));
//         for (int i = 0; i < num_test_; ++i)
//         {
//             std::invoke(func, std::forward<Args>(args)...);
//         }
//         CHECK(cudaEventRecord(stop, 0));
//         CHECK(cudaEventSynchronize(stop));

//         float duration;
//         CHECK(cudaEventElapsedTime(&duration, start, stop));
//         return duration / num_test_ * 1000.0f;
//     }
// };