#pragma once

#include <curand_kernel.h>

const bool debug = false;

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

const int rays_per_pixel = 40;
const int disk_light_shadow_check = 20;

// typedef curandStateMRG32k3a RandState;
// typedef curandStatePhilox4_32_10_t RandState;
// typedef curandStateSobol32_t RandState;
typedef curandState_t RandState;

const int rand_seed = 2021;

const int curve_resolution = 30;
const int angle_steps = 40; // angle split
const float angle_step = 2.f * M_PIf32 / (float)angle_steps;

const int repeat_limit = 1;
const int iterate_limit = 20;

const int max_BSP_depth = 12;
const int max_BSP_leaf_size = 68;
