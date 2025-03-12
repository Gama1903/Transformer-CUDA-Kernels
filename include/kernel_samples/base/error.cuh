#pragma once

#include <iostream>

#define CHECK(call)                                                                  \
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
