#pragma once

#include <vecmath.h>

#include <cfloat>
#include <cmath>

#include "ray.h"

#include "cuda_error.h"

class Camera {
public:
    Camera(const Vector3f &pos, const Vector3f &direction, const Vector3f &up, int imgW, int imgH);

    virtual ~Camera() = default;

    // Generate rays for each screen-space coordinate
    __device__ virtual Ray *generateRay(const Vector2f &point) = 0;

    __host__ __device__ inline int getWidth() const { return width; }

    __host__ __device__ inline int getHeight() const { return height; }

protected:
    // Extrinsic parameters
    Vector3f pos;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;

    // Intrinsic parameters
    int width;
    int height;
};
