#pragma once

#include <vecmath.h>

#include "object3d.h"

class Triangle : public Object3D {
public:
    __device__ Triangle() = delete;

    // a b c are three vertex positions of the triangle
    __device__ Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                        Material *m);

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              RandState &rand_state) override;

    __device__ inline void setNormal(const Vector3f &n) { normal = n; }

protected:
    Vector3f normal;
    Vector3f vertices[3];

    Vector3f E1;
    Vector3f E2;
};
