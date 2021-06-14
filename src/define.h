#pragma once

#include <curand_kernel.h>

const bool debug = false;

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

// typedef curandStateMRG32k3a RandState;
typedef curandStatePhilox4_32_10_t RandState;
// typedef curandStateSobol32_t RandState;

const int rand_seed = 2021;
