#pragma once

#include "device/random.h"
#include "point_light.h"

#include <vecmath.h>

class DiskLight : public PointLight {
public:
    __device__ DiskLight() = delete;

    // normal should be an unit vector
    __device__ DiskLight(const Vector3f &position, const Vector3f &color,
                         const Vector3f &normal, const float radius, Type type = DISK);

    __device__ ~DiskLight() override;

    __device__ virtual void getIllumination(const Vector3f &p, Vector3f &dir,
                                            Vector3f &col) override;

private:
    Vector3f x;
    Vector3f y;

    Random random;
};
