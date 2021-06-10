#include "BsplineCurve.h"

BsplineCurve::BsplineCurve(const std::vector<Vector3f> &points) : Curve(points) {
    if (points.size() < 4) {
        printf("Number of control points of BspineCurve must be more than 4!\n");
        exit(0);
    }

    // PA3: add knots
    n = num_controls - 1;

    checkCudaErrors(cudaMallocManaged(&knots, (n + k + 2) * sizeof(float)));

    double step = 1. / (n + k + 1);
    for (int i = 0; i <= n + k + 1; i++) {
        knots[i] = i * step;
    }

    // PA3: B[p][q - k + p] = B_{q, p}(t)
    checkCudaErrors(cudaMallocManaged(&B, (k + 1) * sizeof(float *)));

    for (int p = 0; p < k + 1; p++) {
        checkCudaErrors(cudaMallocManaged(&B[p], (n + 2 - k + p) * sizeof(float)));
    }
}

BsplineCurve::~BsplineCurve() {
    checkCudaErrors(cudaFree(knots));

    // delete B
    for (int i = 0; i < k + 1; i++) {
        checkCudaErrors(cudaFree(B[i]));
    }
    checkCudaErrors(cudaFree(B));
}

__device__ CurvePoint BsplineCurve::curve_point_at_t(float t) {
    // calculate B[p][q - k + p] = B_{q, p}(t)
    for (int q = k; q <= n + 1; q++) {
        B[0][q - k] = (knots[q] <= t && t < knots[q + 1]) ? 1.f : 0.f;
    }
    for (int p = 1; p <= k; p++) {
        B[p][0] = (knots[k + 1] - t) / (knots[k + 1] - knots[k - p + 1]) * B[p - 1][0];
        for (int q = k - p + 1; q <= n; q++) {
            B[p][q - k + p] =
                (t - knots[q]) / (knots[q + p] - knots[q]) * B[p - 1][q - k + p - 1] +
                (knots[q + p + 1] - t) / (knots[q + p + 1] - knots[q + 1]) * B[p - 1][q - k + p];
        }
        B[p][n + 1 - k + p] =
            (t - knots[n + 1]) / (knots[n + 1 + p] - knots[n + 1]) * B[p - 1][n - k + p];
    }

    Vector3f V = Vector3f(0.f);
    for (int q = 0; q <= n; q++) {
        V += B[k][q] * controls[q];
    }

    Vector3f T = Vector3f(0.f);
    int q = 0;
    T += controls[q] * k *
         (0 / (knots[q + k] - knots[q]) - B[k - 1][q] / (knots[q + k + 1] - knots[q + 1]));
    for (q = 1; q < n; q++) {
        T += controls[q] * k *
             (B[k - 1][q - 1] / (knots[q + k] - knots[q]) -
              B[k - 1][q] / (knots[q + k + 1] - knots[q + 1]));
    }

    return {V, T};
}

/*__device__ void BsplineCurve::discretize(int resolution, std::vector<CurvePoint> &data) {
    data.clear();
    // PA3: fill in data vector
    for (int i = k; i < n + 1; i++) {
        float step = (knots[i + 1] - knots[i]) / float(resolution);
        for (int j = 0; j <= resolution; j++) {
            float t = knots[i] + j * step;
            data.push_back(curve_point_at_t(t));
        }
    }
}
*/
