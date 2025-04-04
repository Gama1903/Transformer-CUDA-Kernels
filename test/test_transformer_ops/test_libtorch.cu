#include <gtest/gtest.h>
#include <base/common.cuh>
#include <torch/torch.h>
#include <iostream>

TEST(LibTorchTest, Test)
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}