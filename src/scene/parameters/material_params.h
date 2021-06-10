#pragma once

#include <vecmath.h>

struct MaterialParams {
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
    float reflect_coefficient;
    float refract_coefficient;
    float refractive_index;
};
