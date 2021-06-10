#pragma once

#include "curve.h"

class BezierCurve : public Curve {
public:
    explicit BezierCurve(const std::vector<Vector3f> &points);

    ~BezierCurve();

    //__device__ void discretize(int resolution, std::vector<CurvePoint> &data) override;

    __device__ CurvePoint curve_point_at_t(float t) override;

private:
    int n;
    double **B;
};
