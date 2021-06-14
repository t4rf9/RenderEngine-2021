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
    float t;

    if (d[0] != 0) {
        t = (min[0] - o[0]) / d[0];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[1] <= p[1] && min[2] <= p[2] && p[1] <= max[1] && p[2] <= max[2]) {
                return true;
            }
        }

        t = (max[0] - o[0]) / d[0];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[1] <= p[1] && min[2] <= p[2] && p[1] <= max[1] && p[2] <= max[2]) {
                return true;
            }
        }
    }

    if (d[1] != 0) {
        t = (min[1] - o[1]) / d[1];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[0] <= p[0] && min[2] <= p[2] && p[0] <= max[0] && p[2] <= max[2]) {
                return true;
            }
        }

        t = (max[1] - o[1]) / d[1];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[0] <= p[0] && min[2] <= p[2] && p[0] <= max[0] && p[2] <= max[2]) {
                return true;
            }
        }
    }

    if (d[2] != 0) {
        t = (min[2] - o[2]) / d[2];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[0] <= p[0] && min[1] <= p[1] && p[0] <= max[0] && p[1] <= max[1]) {
                return true;
            }
        }

        t = (max[2] - o[2]) / d[2];
        if (t > t_min) {
            Vector3f p = o + d * t;
            if (min[0] <= p[0] && min[1] <= p[1] && p[0] <= max[0] && p[1] <= max[1]) {
                return true;
            }
        }
    }

    return false;
}
