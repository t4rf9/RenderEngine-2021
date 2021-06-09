#include "curve.h"

Curve::Curve(std::vector<Vector3f> points) : controls(std::move(points)) {
    Vector3f min = controls[0];
    Vector3f max = controls[0];

    for (int i = 1; i < controls.size(); i++) {
        const Vector3f &p = controls[i];

        for (int j = 0; j < 3; j++) {
            if (p[j] < min[j]) {
                min[j] = p[j];
            }
            if (p[j] > max[j]) {
                max[j] = p[j];
            }
        }
    }

    pBox = new BoundingBox(min, max);
}

Curve::~Curve() { delete pBox; }
