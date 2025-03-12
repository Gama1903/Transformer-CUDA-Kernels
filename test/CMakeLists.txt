# test文件夹cmake文件

# 查找第三方库
find_package(GTest COMPONENTS gtest CONFIG REQUIRED)
find_package(glog COMPONENTS glog CONFIG REQUIRED)

# 添加可执行文件
add_executable(test_kernel_samples)
file(GLOB_RECURSE TEST_SOURCE CONFIGURE_DEPENDS *.cpp *.cu)
target_sources(test_kernel_samples PUBLIC ${TEST_SOURCE})

# 包含头文件
target_include_directories(test_kernel_samples PUBLIC ../include)

# 链接库文件
target_link_libraries(test_kernel_samples PUBLIC GTest::gtest glog::glog)

# 测试设置
enable_testing()
add_test(NAME test_kernel_samples COMMAND test_kernel_samples)