# 根文件夹cmake文件，配置基本设置

# CMAKE最低版本要求
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# 设置CMAKE内置变量
# CPP设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80" "86")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# # libtorch设置
# set(Torch_DIR /usr/local/lib/python3.10/dist-packages/libtorch/share/cmake/Torch)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 其他设置
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 项目信息
project(kernel_samples VERSION 0.1.0 LANGUAGES C CXX CUDA)

# 默认设置
# 未指定构建类型时，默认为Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 未指定库类型时，默认为动态库
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# 添加编译选项
if(CMAKE_CXX_STANDARD LESS 17)
    add_compile_options(-lstdc++fs)
endif()

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 添加子目录
add_subdirectory(test)