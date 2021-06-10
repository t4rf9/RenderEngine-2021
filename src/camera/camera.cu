#include "camera.h"

Camera::Camera(const Vector3f &pos, const Vector3f &direction, const Vector3f &up, int imgW,
               int imgH)
    : pos(pos), direction(direction.normalized()), up(up.normalized()), width(imgW), height(imgH) {
    this->horizontal = Vector3f::cross(this->direction, this->up);
}
