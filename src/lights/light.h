#pragma once

#include <Vector3f.h>

#include "cuda_error.h"

class Light {
public:
    Light() = default;

    virtual ~Light() = default;

    __device__ virtual void getIllumination(const Vector3f &p, Vector3f &dir,
                                            Vector3f &col) const = 0;
};
