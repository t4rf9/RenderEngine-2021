#pragma once

#include <curand_kernel.h>

const bool debug = false;

const bool shadow = false;
const bool reflect = false;
const bool refract = false;

const int rays_per_pixel = 1;
const int disk_light_shadow_check = 1;

// typedef curandStateMRG32k3a RandState;
// typedef curandStatePhilox4_32_10_t RandState;
// typedef curandStateSobol32_t RandState;
typedef curandState_t RandState;

const int rand_seed = 2021;

const int curve_resolution = 60;
const int angle_steps = 60; // angle split
const float angle_step = 2.f * float(M_PI) / (float)angle_steps;

const int repeat_limit = 10;
const int iterate_limit = 100;
