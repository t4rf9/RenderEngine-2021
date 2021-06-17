#pragma once

#include <cmath>
#include <utility>

#include "camera.h"
#include "device/random.h"

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
                                 const Vector3f &up, int imgW, int imgH, float angle,
                                 float focus_dst = 1.f, float aperture = 0.f);

    __device__ Ray generateRay(const Vector2f &point) override;

private:
    __device__ Vector2f random_unit_circle();

    Random random;

protected:
    float angle;
    Vector2f center;

    float focus_dst;
    float aperture;

    float f;
    Matrix3f R;
};