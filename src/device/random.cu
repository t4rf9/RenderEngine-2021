#include "random.h"

__device__ Random::Random() : x(std::default_random_engine::default_seed) {}

__device__ Random::Random(unsigned x) : x(x) {}

__device__ float Random::operator()() {
    x *= 48217;
    x %= 2147483647;
    return x / 2147483647.;
}