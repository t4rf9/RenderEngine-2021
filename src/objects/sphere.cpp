#include "sphere.h"

Sphere::Sphere() : center(0), radius(1), radius_squared(1) {}

Sphere::Sphere(const Vector3f &center, float radius, Material *material)
    : Object3D(material), center(center), radius(radius), radius_squared(radius * radius) {}

bool Sphere::intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) {
    // origin lies anywhere
    Vector3f l = center - ray.getOrigin();
    bool inside = l.length() < radius;
    bool on = l.length() == radius;

    float tp = Vector3f::dot(l, ray.getDirection());
    if (!inside && tp <= 0) {
        return false;
    }

    float d_squared = l.squaredLength() - tp * tp;
    if (d_squared > radius_squared) {
        return false;
    }

    float dt = std::sqrt(radius_squared - d_squared);

    float t = tp + ((inside || on) ? dt : -dt);
    if (t <= t_min || t > hit.getT()) {
        return false;
    }

    Vector3f intersection = ray.pointAtParameter(t);
    Vector3f normal = intersection - center;
    normal.normalize();

    hit.set(t, material, normal);

    return true;
}
