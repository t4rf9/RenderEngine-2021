#pragma once

#include "device/hit.h"
#include "device/ray.h"

class BoundingObject {
public:
    __device__ virtual bool intersect(const Ray &ray, float t_min) = 0;

private:
};
