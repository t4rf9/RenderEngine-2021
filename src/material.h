#pragma once

#include "hit.h"
#include "ray.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vecmath.h>

class Material {
public:
    explicit Material(const Vector3f &d_color, const Vector3f &s_color, float shininess,
                      float reflect_coefficient, float refract_coefficient, float refractive_index);

    virtual ~Material() = default;

    virtual Vector3f getSpecularColor() const;

    virtual Vector3f getDiffuseColor() const;

    Vector3f Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight,
                   const Vector3f &lightColor);

    inline bool reflective() const { return reflect_coefficient > 0; }

    inline bool refractive() const { return refract_coefficient > 0; }

    inline float get_reflect_coefficient() const { return reflect_coefficient; }

    inline float get_refract_coefficient() const { return refract_coefficient; }

    inline float get_refractive_index() const { return refractive_index; }

private:
    inline float clamp(float f) const { return f > 0 ? f : 0; }

protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
    float reflect_coefficient;
    float refract_coefficient;
    float refractive_index;
};
