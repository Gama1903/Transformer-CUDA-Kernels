#pragma once
#include <stdio.h>

__global__ void kernel()
{
    printf("Hello, CUDA!\n");
}