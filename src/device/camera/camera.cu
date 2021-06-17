#include "camera.h"

__device__ Camera::Camera(const Vector3f &pos, const Vector3f &direction,
                          const Vector3f &up, int imgW, int imgH)
    : pos(pos), direction(direction.normalized()), width(imgW), height(imgH) {
    this->horizontal = Vector3f::cross(direction, up).normalized();
    this->up = Vector3f::cross(horizontal, this->direction);
}

__device__ Camera::~Camera() {}
