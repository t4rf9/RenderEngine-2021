#include "random.h"

__device__ Random::Random() : x(1) {}

__device__ Random::Random(uint_fast32_t x) : x(x) {}

__device__ float Random::operator()() {
    x *= 16807;
    x %= 2147483647;
    return x / 2147483647.f;
}