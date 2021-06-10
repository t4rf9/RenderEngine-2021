#include "directional_light.h"

__device__ DirectionalLight::DirectionalLight(const Vector3f &d, const Vector3f &c)
    : direction(d.normalized()), color(c) {}

__device__ void DirectionalLight::getIllumination(const Vector3f &p, Vector3f &dir,
                                                  Vector3f &col) const {
    // the direction to the light is the opposite of the
    // direction of the directional light source
    dir = -direction;
    col = color;
}