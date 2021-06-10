#pragma once

#include "light.h"

#include <Vector3f.h>

class DirectionalLight : public Light {
public:
    __device__ DirectionalLight() = delete;

    __device__ DirectionalLight(const Vector3f &d, const Vector3f &c);

    __device__ ~DirectionalLight() override = default;

    /**
     * @param p unsed in this function
     * @param distanceToLight not well defined because it's not a point light
     */
    __device__ void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override;

private:
    Vector3f direction;
    Vector3f color;
};
