#pragma once

#include <vecmath.h>

#include "axis.h"
#include "object3d.h"

class Triangle : public Object3D {
public:
    __device__ Triangle() = delete;

    // a b c are three vertex positions of the triangle
    __device__ Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                        Material *m, int id = -1);

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              RandState &rand_state) override;

    __device__ bool intersect(const Ray &ray, float t_min, float t_max,
                              RandState &rand_state) override;

    __device__ inline void setNormal(const Vector3f &n) { normal = n; }

    __device__ inline Vector3f getCenter() const { return center; }

    __device__ inline bool intersect_plane(Axis axis, float value) const {
        return min[axis] <= value && value <= max[axis];
    }

protected:
    Vector3f normal;
    Vector3f vertices[3];
    Vector3f center;
    Vector3f min;
    Vector3f max;

    Vector3f E1;
    Vector3f E2;

    int id;
};
