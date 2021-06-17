#pragma once

#include <Vector3f.h>

#include "cuda_error.h"

class Light {
public:
    enum Type { POINT, DISK, DIRECTIONAL } type;

    __device__ Light() = delete;

    __device__ Light(Type type);

    __device__ virtual ~Light();

    __device__ virtual void getIllumination(const Vector3f &p, Vector3f &dir,
                                            Vector3f &col) = 0;
};
