#pragma once

#include <vecmath.h>

#include "BoundingObject.h"

class BoundingBox : public BoundingObject {
public:
    __device__ BoundingBox() = delete;

    __device__ BoundingBox(const Vector3f &min, const Vector3f &max);

    __device__ bool intersect(const Ray &ray, float t_min) override;

    __device__ inline const Vector3f &get_min() { return min; }

    __device__ inline const Vector3f &get_max() { return max; }

private:
    Vector3f min;
    Vector3f max;
};