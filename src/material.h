#pragma once

#include "hit.h"
#include "ray.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vecmath.h>

#include "cuda_error.h"

class Material {
public:
    explicit Material(const Vector3f &d_color, const Vector3f &s_color, float shininess,
                      float reflect_coefficient, float refract_coefficient, float refractive_index);

    virtual ~Material() = default;

    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

    __device__ virtual Vector3f getSpecularColor() const;

    __device__ virtual Vector3f getDiffuseColor() const;

    __device__ Vector3f Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight,
                              const Vector3f &lightColor);

    __device__ inline bool reflective() const { return reflect_coefficient > 0.f; }

    __device__ inline bool refractive() const { return refract_coefficient > 0.f; }

    __device__ inline float get_reflect_coefficient() const { return reflect_coefficient; }

    __device__ inline float get_refract_coefficient() const { return refract_coefficient; }

    __device__ inline float get_refractive_index() const { return refractive_index; }

private:
    __device__ inline float clamp(float f) const { return f > 0.f ? f : 0.f; }

protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
    float reflect_coefficient;
    float refract_coefficient;
    float refractive_index;
};
