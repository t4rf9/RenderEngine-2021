#include "plane.h"

Plane::Plane() {}

Plane::Plane(const Vector3f &normal, float d, Material *m) : Object3D(m), normal(normal), d(-d) {
    float norm = this->normal.normalize();
    this->d /= norm;
}

bool Plane::intersect(const Ray &r, Hit &h, float tmin, uint_fast32_t &rand) {
    float t = -(d + Vector3f::dot(normal, r.getOrigin())) / Vector3f::dot(normal, r.getDirection());
    if (t <= tmin || t > h.getT()) {
        return false;
    }
    h.set(t, material, normal);
    return true;
}
