#pragma once

#include "curve.h"

class BezierCurve : public Curve {
public:
    explicit BezierCurve(const std::vector<Vector3f> &points);

    ~BezierCurve();

    void discretize(int resolution, std::vector<CurvePoint> &data) override;

    CurvePoint curve_point_at_t(double t) override;

private:
    int n;
    double **B;
};
