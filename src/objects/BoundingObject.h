#pragma once

#include "hit.h"
#include "ray.h"
#include "cuda_error.h"

class BoundingObject {
public:
    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

    __device__ virtual bool intersect(const Ray &ray, float t_min) = 0;

private:
};
