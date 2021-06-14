#pragma once

#include <curand_kernel.h>

#include "device/random.h"

const bool debug = false;

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

// typedef curandStateMRG32k3a RandState;
// typedef curandStatePhilox4_32_10_t RandState;
// typedef curandStateSobol32_t RandState;

typedef Random RandState;

const int rand_seed = 2021;
