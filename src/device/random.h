#pragma once

#include <vecmath.h>

#include <random>

class Random {
public:
    __device__ Random() : x(1) {}

    __device__ explicit Random(uint_fast32_t x) : x(x) {}

    __device__ inline float operator()() {
        x *= 16807;
        x %= 2147483647;
        return x / 2147483647.f;
    }

    __device__ Vector2f unit_disk();

private:
    uint_fast32_t x;
};
