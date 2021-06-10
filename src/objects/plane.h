#pragma once

#include <vecmath.h>

#include <cmath>

#include "object3d.h"

// PA1: Implement Plane representing an infinite plane

class Plane : public Object3D {
public:
    Plane();

    Plane(const Vector3f &normal, float d, Material *m);

    ~Plane() override = default;

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState *rand_state) override;

protected:
    Vector3f normal; // (a, b, c)
    float d;         // ax + by + cz + d = 0
};
