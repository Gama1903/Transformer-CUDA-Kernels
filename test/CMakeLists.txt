# test文件夹cmake文件

# 查找第三方库
find_package(GTest COMPONENTS gtest CONFIG REQUIRED)
find_package(glog COMPONENTS glog CONFIG REQUIRED)

# find_package(Torch REQUIRED)

# 添加可执行文件
add_executable(test)
file(GLOB_RECURSE TEST_SOURCE CONFIGURE_DEPENDS *.cpp *.cu)
target_sources(test PUBLIC ${TEST_SOURCE})

# 调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G> # 启用CUDA调试

        # -DALIGNED_DATA # 启用对齐数据
    )
endif()

# 包含头文件
target_include_directories(test PUBLIC ../include ${CUDAToolkit_INCLUDE_DIRS})

# 链接库文件
target_link_libraries(test PUBLIC GTest::gtest glog::glog CUDA::cudart CUDA::cublas

    # ${TORCH_LIBRARIES}
)

# 测试设置
enable_testing()
add_test(NAME test COMMAND test)