#pragma once

#include <curand_kernel.h>

#include "cuda_error.h"
#include "hit.h"
#include "material.h"
#include "ray.h"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D();

    virtual ~Object3D() = default;

    explicit Object3D(Material *material);

    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

    // Intersect Ray with this object. If hit, store information in hit structure.
    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      curandState *rand_state) = 0;

protected:
    Material *material;
};
