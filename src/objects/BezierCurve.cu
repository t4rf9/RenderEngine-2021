#include "BezierCurve.h"

BezierCurve::BezierCurve(const std::vector<Vector3f> &points) : Curve(points) {
    if (points.size() < 4 || points.size() % 3 != 1) {
        printf("Number of control points of a BezierCurve must be 3n+1!\n");
        exit(0);
    }

    checkCudaErrors(cudaMallocManaged(&B, num_controls * sizeof(float *)));
    for (int i = 0; i < num_controls; i++) {
        checkCudaErrors(cudaMallocManaged(&B[i], (i + 1) * sizeof(float)));
    }
}

BezierCurve::~BezierCurve() {
    for (int i = 0; i <= n; i++) {
        checkCudaErrors(cudaFree(B[i]));
    }
    checkCudaErrors(cudaFree(B));
}

__device__ CurvePoint BezierCurve::curve_point_at_t(float t) {
    // calculate B[p][q] = B_{q, p}(t)
    B[0][0] = 1.f;
    for (int p = 1; p <= n; p++) {
        B[p][0] = (1 - t) * B[p - 1][0];
        for (int q = 1; q < p; q++) {
            B[p][q] = t * B[p - 1][q - 1] + (1 - t) * B[p - 1][q];
        }
        B[p][p] = t * B[p - 1][p - 1];
    }

    Vector3f V = Vector3f(0.f);
    for (int j = 0; j <= n; j++) {
        V += B[n][j] * controls[j];
    }

    Vector3f T = Vector3f(0.f);
    for (int j = 0; j < n; j++) {
        T += B[n - 1][j] * (controls[j + 1] - controls[j]);
    }
    T *= n;

    return {V, T};
}

/*
__device__ void BezierCurve::discretize(int resolution, std::vector<CurvePoint> &data) {
    data.clear();
    // PA3: fill in data vector

    double step = 1. / resolution;
    for (int i = 0; i <= resolution; i++) {
        double t = i * step;
        data.push_back(curve_point_at_t(t));
    }
}
*/
