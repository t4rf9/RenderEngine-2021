#include "perspective_camera.h"

PerspectiveCamera::PerspectiveCamera(const Vector3f &pos, const Vector3f &direction,
                                     const Vector3f &up, int imgW, int imgH, float angle)
    : Camera(pos, direction, up, imgW, imgH), angle(angle),
      center(float(imgW) / 2, float(imgH) / 2), f(2 * std::tan(angle / 2) / float(imgH)),
      R(this->horizontal, this->up, this->direction) {}

Ray PerspectiveCamera::generateRay(const Vector2f &point) {
    float dx = point[0] - center[0];
    float dy = point[1] - center[1];
    Vector3f d_camera(dx * f, dy * f, 1);

    return Ray(pos, (R * d_camera).normalized(), 0, 1, 1);
}
