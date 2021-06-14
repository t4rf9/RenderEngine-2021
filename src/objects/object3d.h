#pragma once

#include "hit.h"
#include "material.h"
#include "ray.h"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D();

    virtual ~Object3D() = default;

    explicit Object3D(Material *material);

    // Intersect Ray with this object. If hit, store information in hit structure.
    virtual bool intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) = 0;

protected:
    Material *material;
};
