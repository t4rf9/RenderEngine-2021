#include "curve.h"

Curve::Curve(const std::vector<Vector3f> &points) : num_controls(points.size()) {
    checkCudaErrors(cudaMallocManaged(&controls, num_controls * sizeof(Vector3f)));

    controls[0] = points[0];

    Vector3f min = controls[0];
    Vector3f max = controls[0];
    for (int i = 1; i < num_controls; i++) {
        Vector3f &p = controls[i];
        p = points[i];
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

Curve::~Curve() {
    checkCudaErrors(cudaFree(controls));
    delete pBox;
}

bool Curve::IsFlat() const {
    for (int i = 0; i < num_controls; i++) {
        if (controls[i].z() != 0.0f) {
            return false;
        }
    }
    return true;
}
