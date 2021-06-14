#include "transform.h"

Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point) {
    return (mat * Vector4f(point, 1)).xyz();
}

Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir) {
    return (mat * Vector4f(dir, 0)).xyz();
}

Transform::Transform(const Matrix4f &m, Object3D *obj) : o(obj) { transform = m.inverse(); }

bool Transform::intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) {
    Vector3f trSource = transformPoint(transform, ray.getOrigin());
    Vector3f trDirection = transformDirection(transform, ray.getDirection());
    float trDirLen = trDirection.normalize();
    Ray tr(trSource, trDirection, ray.get_depth(), ray.get_weight(),
           ray.get_incident_refractive_index());
    bool inter = o->intersect(tr, hit, t_min, rand);
    if (inter) {
        // n^T t = 0
        // => [n; 0]^T [t; 0] = 0
        // => [n; 0]^T M^{-1} [t'; 0] = 0
        // => [n'; 0] = (M^{-1})^T [n; 0]
        hit.set(hit.getT() / trDirLen, hit.getMaterial(),
                transformDirection(transform.transposed(), hit.getNormal()).normalized());
    }
    return inter;
}
