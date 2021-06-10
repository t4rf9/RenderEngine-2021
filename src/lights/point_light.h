#pragma once

#include "light.h"

#include <Vector3f.h>

class PointLight : public Light {
public:
    PointLight() = delete;

    PointLight(const Vector3f &p, const Vector3f &c);

    ~PointLight() override = default;

    __device__ void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override;

private:
    Vector3f position;
    Vector3f color;
};
