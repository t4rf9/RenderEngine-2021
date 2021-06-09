#include "BoundingBox.h"

BoundingBox::BoundingBox(const Vector3f &min, const Vector3f &max) : min(min), max(max) {
    // std::cout << "min:\t(" << min[0] << ", " << min[1] << ", " << min[2] << ")" << std::endl;
    // std::cout << "max:\t(" << max[0] << ", " << max[1] << ", " << max[2] << ")\n" << std::endl;
}

bool BoundingBox::intersect(const Ray &ray, float t_min) {
    const Vector3f &d = ray.getDirection();
    const Vector3f &o = ray.getOrigin();

    /*
    if (min <= o && o <= max) {
        return true;
    }
    */

    // std::cout << "min" << std::endl;
    for (int i = 0; i < 3; i++) {
        if (d[i] != 0) {
            float t = (min[i] - o[i]) / d[i];
            if (t > t_min) {
                Vector3f p = ray.pointAtParameter(t);
                // std::cout << "\tp:\t(" << p[0] << ", " << p[1] << ", " << p[2] << ")" <<
                // std::endl;
                if (min <= p && p <= max) {
                    return true;
                }
            }
        }
    }

    // std::cout << "max" << std::endl;
    for (int i = 0; i < 3; i++) {
        if (d[i] != 0) {
            float t = (max[i] - o[i]) / d[i];
            if (t > t_min) {
                Vector3f p = ray.pointAtParameter(t);
                // std::cout << "\tp:\t(" << p[0] << ", " << p[1] << ", " << p[2] << ")" <<
                // std::endl;
                if (min <= p && p <= max) {
                    return true;
                }
            }
        }
    }

    return false;
}
