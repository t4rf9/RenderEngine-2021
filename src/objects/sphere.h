#pragma once

#include <vecmath.h>

#include <cmath>

#include "object3d.h"

class Sphere : public Object3D {
public:
    Sphere();

    Sphere(const Vector3f &center, float radius, Material *material);

    ~Sphere() override = default;

    bool intersect(const Ray &ray, Hit &hit, float t_min) override;

protected:
    Vector3f center;
    float radius;
    float radius_squared;
};
