#include "directional_light.h"

__device__ DirectionalLight::DirectionalLight(const Vector3f &direction,
                                              const Vector3f &color, Type type)
    : Light(type), direction(direction.normalized()), color(color) {}

__device__ DirectionalLight::~DirectionalLight() {}

__device__ void DirectionalLight::getIllumination(const Vector3f &p, Vector3f &dir,
                                                  Vector3f &col, RandState &random) {
    // the direction to the light is the opposite of the
    // direction of the directional light source
    dir = -direction;
    col = color;
}