#pragma once

#include "define.h"

#include <curand_kernel.h>

#include "device/hit.h"
#include "device/material.h"
#include "device/ray.h"

// Base class for all 3d entities.
class Object3D {
public:
    __device__ Object3D();

    __device__ virtual ~Object3D();

    __device__ explicit Object3D(Material *material);

    // Intersect Ray with this object. If hit, store information in hit structure.
    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state) = 0;

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state) = 0;

protected:
    Material *material;
};
