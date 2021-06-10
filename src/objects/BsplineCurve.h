#pragma once

#include "curve.h"

#include "cuda_error.h"

class BsplineCurve : public Curve {
public:
    explicit BsplineCurve(const std::vector<Vector3f> &points);

    ~BsplineCurve();

    //__device__ void discretize(int resolution, std::vector<CurvePoint> &data) override;

    __device__ CurvePoint curve_point_at_t(float t) override;

protected:
    int n;
    int k = 3;
    float *knots;

private:
    float **B;
};
