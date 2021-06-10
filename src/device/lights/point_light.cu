#include "point_light.h"

__device__ PointLight::PointLight(const Vector3f &position, const Vector3f &color)
    : position(position), color(color) {}

__device__ void PointLight::getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const {
    // the direction to the light is the opposite of the
    // direction of the directional light source
    dir = (position - p);
    col = color;
}