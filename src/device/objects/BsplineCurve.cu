#include "BsplineCurve.h"

__device__ BsplineCurve::BsplineCurve(Vector3f *points, int num_controls)
    : Curve(points, num_controls) {
    /*
    if (points.size() < 4) {
        printf("Number of control points of BspineCurve must be more than 4!\n");
        exit(0);
    }
    */

    // PA3: add knots
    n = num_controls - 1;

    knots = new float[n + k + 2];

    double step = 1. / (n + k + 1);
    for (int i = 0; i <= n + k + 1; i++) {
        knots[i] = i * step;
    }
}

__device__ BsplineCurve::~BsplineCurve() { delete[] knots; }

__device__ CurvePoint BsplineCurve::curve_point_at_t(float t) {
    __shared__ extern float shared[];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    float *B = &shared[thread_id * (k + 1) * (2 * n + 4 - k) / 2];

    // PA3: B[p][q - k + p] = B_{q, p}(t)
    // B[p][r] = B[p * (2 * n + 3 - 2 * k + p) / 2 + r]
    //         = B[p * (p + pp) / 2 + r]
    int pp = 2 * n + 3 - 2 * k;

    // calculate B[p][q - k + p] = B_{q, p}(t)
    for (int q = k; q <= n + 1; q++) {
        B[0 * (0 + pp) / 2 + q - k] = (knots[q] <= t && t < knots[q + 1]) ? 1.f : 0.f;
    }
    for (int p = 1; p <= k; p++) {
        int offset_p = p * (p + pp) / 2;
        int offset_p_1 = (p - 1) * (p - 1 + pp) / 2;
        B[offset_p + 0] =
            (knots[k + 1] - t) / (knots[k + 1] - knots[k - p + 1]) * B[offset_p_1 + 0];
        for (int q = k - p + 1; q <= n; q++) {
            B[offset_p + q - k + p] = (t - knots[q]) / (knots[q + p] - knots[q]) *
                                          B[offset_p_1 + q - k + p - 1] +
                                      (knots[q + p + 1] - t) /
                                          (knots[q + p + 1] - knots[q + 1]) *
                                          B[offset_p_1 + q - k + p];
        }
        B[offset_p + n + 1 - k + p] = (t - knots[n + 1]) /
                                      (knots[n + 1 + p] - knots[n + 1]) *
                                      B[offset_p_1 + n - k + p];
    }

    int offset_k = k * (k + pp) / 2;
    Vector3f V = Vector3f(0.f);
    for (int q = 0; q <= n; q++) {
        V += B[offset_k + q] * controls[q];
    }

    int q = 0;
    int offset_k_1 = (k - 1) * (k - 1 + pp) / 2;
    Vector3f T = -controls[q] * k * B[offset_k_1 + q] / (knots[q + k + 1] - knots[q + 1]);
    for (q = 1; q < n; q++) {
        T += controls[q] * k *
             (B[offset_k_1 + q - 1] / (knots[q + k] - knots[q]) -
              B[offset_k_1 + q] / (knots[q + k + 1] - knots[q + 1]));
    }

    return {V, T};
}

/* void BsplineCurve::discretize(int resolution, std::vector<CurvePoint> &data) {
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
