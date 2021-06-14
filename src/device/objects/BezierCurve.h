#pragma once

#include "curve.h"

class BezierCurve : public Curve {
public:
    __device__ explicit BezierCurve(Vector3f *points, int num_controls);

    __device__ ~BezierCurve();

    // void discretize(int resolution, std::vector<CurvePoint> &data) override;

    __device__ CurvePoint curve_point_at_t(float t) override;

private:
    int n;
    float **B;
};
