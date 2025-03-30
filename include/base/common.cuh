#pragma once

#include <cublas_v2.h> // cublasHandle_t
#include <cstdio>      // snprintf
#include <stdexcept>   // runtime_error
#include <cuda_fp16.h> // half
#include <type_traits> // 类型萃取
#include <random>      // 随机数生成

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

// 最大数据长度
constexpr int N = 1024 * 1024 * 32;

// CUDA常量
constexpr int block_size = 256;
constexpr int warp_size = 32;
constexpr int num_wave = 32;

// 获取grid_size
__forceinline__ cudaError_t getGridSize(int n, int *grid_size, int block_size = block_size, int num_wave = num_wave)
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

// CUDA基准测试类
struct CudaBenchmark
{
    struct Result
    {
        float avg_time;
        float total_time;
    };

    int num_runs_;
    int num_warmup_;

    explicit CudaBenchmark(int num_runs = 50, int num_warmup = 2) : num_runs_(num_runs), num_warmup_(num_warmup) {}

    template <class SetupFunc, class KernelFunc>
    Result run(SetupFunc &&setup, KernelFunc &&kernel)
    {
        for (int i = 0; i < num_warmup_; i++)
        {
            setup();
            kernel();
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        Result result;
        for (int i = 0; i < num_runs_; i++)
        {
            setup();
            CHECK_CUDA(cudaEventRecord(start));
            kernel();
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float elapsed_time;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
            result.total_time += elapsed_time;
        }
        result.avg_time = result.total_time / num_runs_;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        return result;
    }

    static void print_result(Result const &result)
    {
        printf("Average time: %f ms\n", result.avg_time);
        // printf("Total time: %f ms\n", result.total_time);
    }
};
