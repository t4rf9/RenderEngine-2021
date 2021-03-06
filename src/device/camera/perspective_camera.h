#pragma once

#include <cmath>
#include <utility>

#include "camera.h"

class PerspectiveCamera : public Camera {
public:
    __device__ PerspectiveCamera() = delete;

    /**
     * @param pos
     * @param direction
     * @param up
     * @param imgW
     * @param imgH
     * @param angle in radian
     */
    __device__ PerspectiveCamera(const Vector3f &pos, const Vector3f &direction,
                                 const Vector3f &up, int imgW, int imgH, float angle);

    __device__ Ray generateRay(const Vector2f &point) override;

protected:
    float angle;
    Vector2f center;
    float f;
    Matrix3f R;
};