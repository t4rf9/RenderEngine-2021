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
                                           Vector3f &col, RandState &random) {
    Vector2f offset = random_unit_disk(random);
    dir = (position + x * offset.x() + y * offset.y() - p);
    col = color;
}

__device__ Vector2f DiskLight::random_unit_disk(RandState &random) {
    float r = sqrt(curand_uniform(&random));
    float theta = 2.f * M_PI * curand_uniform(&random);
    return Vector2f(r * cos(theta), r * sin(theta));
}
