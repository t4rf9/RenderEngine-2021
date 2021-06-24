#include "plane.h"

__device__ Plane::Plane() {}

__device__ Plane::Plane(const Vector3f &normal, float d, Material *m)
    : Object3D(m), normal(normal), d(-d) {
    float norm = this->normal.normalize();
    this->d /= norm;
}

__device__ Plane::Plane(const Vector3f &normal, float d, Material *m,
                        const Vector3f &texture_origin, const Vector3f &texture_x,
                        const Vector3f &texture_y)
    : Object3D(m), normal(normal), d(-d), texture_origin(texture_origin),
      texture_x(texture_x), texture_y(texture_y) {
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
    if (material->useTexture()) {
        Vector3f p = ray.pointAtParameter(t);
        Vector3f vec = p - texture_origin;
        int x = Vector3f::dot(vec, texture_x);
        int y = Vector3f::dot(vec, texture_y);
        hit.set(t, material->getTextureColor(x, y));
    } else {
        hit.set(t, material, normal);
    }
    return true;
}

__device__ bool Plane::intersect(const Ray &ray, float t_min, float t_max,
                                 RandState &rand_state) {
    float t = -(d + Vector3f::dot(normal, ray.getOrigin())) /
              Vector3f::dot(normal, ray.getDirection());
    return t_min < t && t < t_max;
}