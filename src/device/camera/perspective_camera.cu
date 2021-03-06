#include "perspective_camera.h"

__device__
PerspectiveCamera::PerspectiveCamera(const Vector3f &pos, const Vector3f &direction,
                                     const Vector3f &up, int imgW, int imgH, float angle)
    : Camera(pos, direction, up, imgW, imgH), angle(angle),
      center(float(imgW / 2), float(imgH / 2)),
      f(2.f * std::tan(angle * 0.5f) / float(imgH)),
      R(this->horizontal, this->up, this->direction) {}

__device__ Ray PerspectiveCamera::generateRay(const Vector2f &point) {
    float dx = point[0] - center[0];
    float dy = point[1] - center[1];
    Vector3f d_camera(dx * f, dy * f, 1.f);

    return Ray(pos, (R * d_camera).normalized(), 0, 1.f, 1.f);
}
