#include <gtest/gtest.h>
#include <glog/logging.h>
#include <filesystem>

int main(int argc, char **argv)
{
    // Initialize Google's logging library.
    google::InitGoogleLogging("kernel_samples");
    ::testing::InitGoogleTest(&argc, argv);

    std::string log_dir = "./log/";
    std::filesystem::create_directories(log_dir); // 创建目录

    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;
    LOG(INFO) << "Start Test...\n";

    return RUN_ALL_TESTS();
}
