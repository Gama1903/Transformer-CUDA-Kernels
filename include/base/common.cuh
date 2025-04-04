#pragma once

#include <cublas_v2.h>   // cublasHandle_t
#include <cstdio>        // snprintf
#include <stdexcept>     // runtime_error
#include <cuda_fp16.h>   // half
#include <type_traits>   // 类型萃取
#include <random>        // 随机数生成
#include <gtest/gtest.h> // gtest

// 向上取整除法
#define CDIV(x, y) (((x) + (y) - 1) / (y))

// CUBLAS错误检查宏
#define CHECK_CUBLAS(call)                                             \
    do                                                                 \
    {                                                                  \
        cublasStatus_t error_code = call;                              \
        if (error_code != CUBLAS_STATUS_SUCCESS)                       \
        {                                                              \
            const char *err_name = cublasGetStatusName(error_code);    \
            const char *err_msg = cublasGetStatusString(error_code);   \
            char buffer[256];                                          \
            snprintf(buffer, sizeof(buffer),                           \
                     "\n [CUBLAS] Error in %s:%d \n %s: %s (code %d)", \
                     __FILE__, __LINE__,                               \
                     err_name, err_msg,                                \
                     static_cast<int>(error_code));                    \
            throw std::runtime_error(buffer);                          \
        }                                                              \
    } while (0)

// CUDA错误检查宏
#define CHECK_CUDA(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t error_code = call;                            \
        if (error_code != cudaSuccess)                            \
        {                                                         \
            const char *err_msg = cudaGetErrorString(error_code); \
            char buffer[256];                                     \
            snprintf(buffer, sizeof(buffer),                      \
                     "\n [CUDA] Error in %s:%d \n %s (code %d)",  \
                     __FILE__, __LINE__, err_msg,                 \
                     static_cast<int>(error_code));               \
            throw std::runtime_error(buffer);                     \
        }                                                         \
    } while (0)

// CUDA常量
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WAVE = 32;

// 获取grid_size
__forceinline__ cudaError_t getGridSize(int n, int *grid_size, int block_size = BLOCK_SIZE, int num_wave = NUM_WAVE)
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

    *grid_size = std::max(1, std::min(
                                 CDIV(n, block_size),
                                 sm_num * thread_per_sm / block_size * num_wave));
    return cudaSuccess;
}

// CUBLAS句柄类
struct CublasHandle
{
    cublasHandle_t handle_;

    CublasHandle() : handle_(nullptr)
    {
        CHECK_CUBLAS(cublasCreate(&handle_));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(CublasHandle const &) = delete;
    CublasHandle &operator=(CublasHandle const &) = delete;

    CublasHandle(CublasHandle &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }
    CublasHandle &operator=(CublasHandle &&other) noexcept
    {
        if (this != &other)
        {
            if (handle_ != nullptr)
            {
                cublasDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    cublasHandle_t get() const { return handle_; }
};

// 类型萃取类
template <class Tp, class = void>
struct TypeTraits;

template <class Tp>
struct TypeTraits<Tp, std::enable_if_t<std::is_arithmetic_v<Tp>>>
{
    static Tp from_float(float f) { return static_cast<Tp>(f); }
    static float to_float(Tp val) { return static_cast<float>(val); }
};

template <>
struct TypeTraits<half>
{
    static half from_float(float f) { return __float2half(f); }
    static float to_float(half val) { return __half2float(val); }
};

// CUDA分配器 统一内存分配
template <class Tp>
struct CudaAllocator
{
    using value_type = Tp;

    Tp *allocate(int size)
    {
        Tp *ptr = nullptr;
        CHECK_CUDA(cudaMallocManaged(&ptr, sizeof(Tp) * size));
        return ptr;
    }

    void deallocate(Tp *ptr, int size = 0)
    {
        CHECK_CUDA(cudaFree(ptr));
    }
};

// CUDA向量类
template <class Tp>
struct CudaVector
{
    using type_traits = TypeTraits<Tp>;

    std::vector<Tp, CudaAllocator<Tp>> vec;

    explicit CudaVector(int size = 0, float value = 0.0f) : vec(size, type_traits::from_float(value)) {}

    explicit CudaVector(int size, float min_val, float max_val, int seed = 42) : vec(size)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(min_val, max_val);
        for (auto &val : vec)
        {
            val = type_traits::from_float(dis(gen));
        }
    }

    CudaVector(const CudaVector &other) noexcept : vec(other.vec) {}
    CudaVector &operator=(const CudaVector &other) noexcept
    {
        if (this != &other)
        {
            vec = other.vec;
        }
        return *this;
    }
    CudaVector(CudaVector &&other) noexcept : vec(std::move(other.vec)) {}
    CudaVector &operator=(CudaVector &&other) noexcept
    {
        if (this != &other)
        {
            vec = std::move(other.vec);
        }
        return *this;
    }

    Tp &operator[](int i) noexcept { return vec[i]; }
    const Tp &operator[](int i) const noexcept { return vec[i]; }

    Tp *data() noexcept { return vec.data(); }
    const Tp *data() const noexcept { return vec.data(); }

    int size() const noexcept { return vec.size(); }

    void print(int n) const
    {
        int size = vec.size();
        if (n >= size)
            throw std::runtime_error("n should be less than size");
        if (n == 1)
            printf("%f\n", type_traits::to_float(vec[0]));
        else
        {
            for (int i = 0; i < n; i++)
                printf("%f ", type_traits::to_float(vec[i]));
            printf("\n");
            for (int i = size - n; i < size; i++)
                printf("%f ", type_traits::to_float(vec[i]));
            printf("\n");
        }
    }
};

// 数值检验函数
inline void verify(float const *res, float const *ref, int N, float tolerance)
{
    int failedIndex = -1;
    for (int i = 0; i < N; i++)
    {
        if (std::abs(res[i] - ref[i]) > tolerance)
        {
            failedIndex = i;
            EXPECT_NEAR(res[i], ref[i], tolerance) << "数值对比失败，索引为: " << failedIndex;
            break;
        }
    }
}

// CUDA基准测试类
struct CudaBenchmark
{
    struct Result
    {
        float avg_time{};
        double total_time{};
        float min_time{std::numeric_limits<float>::max()};
        float max_time{0};
    };

    int num_runs_;
    int num_warmup_;
    std::vector<cudaEvent_t> start_events; // 每个run独立事件
    std::vector<cudaEvent_t> stop_events;

    explicit CudaBenchmark(int num_runs = 1000, int num_warmup = 100)
        : num_runs_(num_runs), num_warmup_(num_warmup)
    {
        // 预创建事件对象
        start_events.resize(num_runs_);
        stop_events.resize(num_runs_);
        for (int i = 0; i < num_runs_; ++i)
        {
            CHECK_CUDA(cudaEventCreate(&start_events[i]));
            CHECK_CUDA(cudaEventCreate(&stop_events[i]));
        }
    }

    ~CudaBenchmark()
    {
        for (auto &e : start_events)
            cudaEventDestroy(e);
        for (auto &e : stop_events)
            cudaEventDestroy(e);
    }

    template <class SetupFunc, class KernelFunc>
    Result run(SetupFunc &&setup, KernelFunc &&kernel)
    {
        // Warmup阶段（无测量）
        setup();
        for (int i = 0; i < num_warmup_; ++i)
        {
            kernel();
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        Result result;
        std::vector<float> measurements;
        measurements.reserve(num_runs_);

        // 阶段1：记录所有事件（无同步）
        setup();
        for (int i = 0; i < num_runs_; ++i)
        {
            CHECK_CUDA(cudaEventRecord(start_events[i])); // 记录启动时间
            kernel();
            CHECK_CUDA(cudaEventRecord(stop_events[i])); // 记录结束时间
        }

        // 阶段2：统一同步所有事件
        for (auto &e : stop_events)
        {
            CHECK_CUDA(cudaEventSynchronize(e)); // 等待所有事件完成
        }

        // 阶段3：计算耗时
        for (int i = 0; i < num_runs_; ++i)
        {
            float elapsed;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed,
                                            start_events[i], stop_events[i]));

            result.total_time += static_cast<double>(elapsed);
            result.min_time = std::min(result.min_time, elapsed);
            result.max_time = std::max(result.max_time, elapsed);
        }

        result.avg_time = static_cast<float>(result.total_time / num_runs_);
        return result;
    }

    void print_result(Result const &result)
    {
        printf("Average time: %f ms\n", result.avg_time);
        // printf("Min time: %f ms\n", result.min_time);
        // printf("Max time: %f ms\n", result.max_time);
    }
};
