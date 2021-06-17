#include "disk_light.h"

__device__ DiskLight::DiskLight(const Vector3f &position, const Vector3f &color,
                                const Vector3f &normal, const float radius, Type type)
    : PointLight(position, color, type) {
    x = Vector3f(-normal.z(), normal.z(),
                 normal.z() != 0 || normal.x() != normal.y() ? normal.x() - normal.y()
                                                             : normal.x() + normal.y())
            .normalized() *
        radius;
    y = Vector3f::cross(normal, x) * radius;
}

__device__ DiskLight::~DiskLight() {}

__device__ void DiskLight::getIllumination(const Vector3f &p, Vector3f &dir,
                                           Vector3f &col) {
    Vector2f offset = random.unit_disk();
    dir = (position + x * offset.x() + y * offset.y() - p);
    col = color;
}
