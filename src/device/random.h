#pragma once

#include <random>

class Random {
public:
    __device__ Random();

    __device__ explicit Random(unsigned x);

    __device__ float operator()();

private:
    uint_fast32_t x;
};
