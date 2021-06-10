#pragma once

#include "hit.h"
#include "ray.h"
#include "cuda_error.h"

class BoundingObject {
public:
    __device__ virtual bool intersect(const Ray &ray, float t_min) = 0;

private:
};
