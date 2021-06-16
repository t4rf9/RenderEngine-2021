#include "BezierCurve.h"

__device__ BezierCurve::BezierCurve(Vector3f *points, int num_controls)
    : Curve(points, num_controls) {
    /*
    if (points.size() < 4 || points.size() % 3 != 1) {
        printf("Number of control points of a BezierCurve must be 3n+1!\n");
        exit(0);
    }
    */

    n = num_controls - 1;
}

__device__ BezierCurve::~BezierCurve() {}

__device__ CurvePoint BezierCurve::curve_point_at_t(float t) {
    __shared__ extern float shared[];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    float *B = &shared[thread_id * (n + 2) * (n + 1) / 2];

    // calculate B[p][q] = B_{q, p}(t)
    // B[p][q] = B[p * (p + 1) / 2 + q]
    B[0 * (0 + 1) / 2 + 0] = 1.f;
    for (int p = 1; p <= n; p++) {
        int offset_p = p * (p + 1) / 2;
        int offset_p_1 = p * (p - 1) / 2;
        B[offset_p + 0] = (1.f - t) * B[offset_p_1 + 0];
        for (int q = 1; q < p; q++) {
            B[offset_p + q] = t * B[offset_p_1 + q - 1] + (1 - t) * B[offset_p_1 + q];
        }
        B[offset_p + p] = t * B[offset_p_1 + p - 1];
    }

    Vector3f V = Vector3f(0.f);
    int offset_n = n * (n + 1) / 2;
    for (int j = 0; j <= n; j++) {
        V += B[offset_n + j] * controls[j];
    }

    int offset_n_1 = n * (n - 1) / 2;
    Vector3f T = Vector3f(0.f);
    for (int j = 0; j < n; j++) {
        T += B[offset_n_1 + j] * (controls[j + 1] - controls[j]);
    }
    T *= n;

    return {V, T};
}

/*
void BezierCurve::discretize(int resolution, std::vector<CurvePoint> &data) {
    data.clear();
    // PA3: fill in data vector

    double step = 1. / resolution;
    for (int i = 0; i <= resolution; i++) {
        double t = i * step;
        data.push_back(curve_point_at_t(t));
    }
}
*/
