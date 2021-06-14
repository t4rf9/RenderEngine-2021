#pragma once

#include "light.h"

#include <Vector3f.h>

class PointLight : public Light {
public:
    __device__ PointLight() = delete;

    __device__ PointLight(const Vector3f &position, const Vector3f &color);

    __device__ ~PointLight() override;

    __device__ void getIllumination(const Vector3f &p, Vector3f &dir,
                                    Vector3f &col) const override;

private:
    Vector3f position;
    Vector3f color;
};
