#include "BoundingBox.h"

__device__ BoundingBox::BoundingBox(const Vector3f &min, const Vector3f &max)
    : min(min), max(max) {}

__device__ bool BoundingBox::intersect(const Ray &ray, float t_min) {
    const Vector3f &d = ray.getDirection();
    const Vector3f &o = ray.getOrigin();

    /*
    if (min <= o && o <= max) {
        return true;
    }
    */

    for (int i = 0; i < 3; i++) {
        if (d[i] != 0) {
            float t = (min[i] - o[i]) / d[i];
            if (t > t_min) {
                Vector3f p = ray.pointAtParameter(t);
                if (min <= p && p <= max) {
                    return true;
                }
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        if (d[i] != 0) {
            float t = (max[i] - o[i]) / d[i];
            if (t > t_min) {
                Vector3f p = ray.pointAtParameter(t);
                if (min <= p && p <= max) {
                    return true;
                }
            }
        }
    }

    return false;
}
