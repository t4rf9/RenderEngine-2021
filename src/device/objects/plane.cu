#include "plane.h"

__device__ Plane::Plane() {}

__device__ Plane::Plane(const Vector3f &normal, float d, Material *m)
    : Object3D(m), normal(normal), d(-d) {
    float norm = this->normal.normalize();
    this->d /= norm;
}

__device__ Plane::~Plane() {}

__device__ bool Plane::intersect(const Ray &ray, Hit &hit, float t_min,
                                 RandState &rand_state) {
    float t = -(d + Vector3f::dot(normal, ray.getOrigin())) /
              Vector3f::dot(normal, ray.getDirection());
    if (t <= t_min || t > hit.getT()) {
        return false;
    }
    hit.set(t, material, normal);
    return true;
}
