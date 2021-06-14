#pragma once

#include <vecmath.h>

#include <cmath>

#include "object3d.h"

class Sphere : public Object3D {
public:
    __device__ Sphere();

    __device__ Sphere(const Vector3f &center, float radius, Material *material);

    __device__ ~Sphere() override;

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState &rand_state) override;

protected:
    Vector3f center;
    float radius;
    float radius_squared;
};
