#include "curve.h"

__device__ Curve::Curve(Vector3f *points, int num_controls) : num_controls(num_controls) {
    // controls = new Vector3f[num_controls];

    // controls[0] = points[0];

    controls = points;

    Vector3f min = controls[0];
    Vector3f max = controls[0];
    for (int i = 1; i < num_controls; i++) {
        Vector3f &p = controls[i];
        // p = points[i];
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

__device__ Curve::~Curve() {
    //delete[] controls; // freed by SceneParser
    delete pBox;
}

__device__ bool Curve::IsFlat() const {
    for (int i = 0; i < num_controls; i++) {
        if (controls[i].z() != 0.0f) {
            return false;
        }
    }
    return true;
}
