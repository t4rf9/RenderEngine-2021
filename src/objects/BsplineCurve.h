#pragma once

#include "curve.h"

class BsplineCurve : public Curve {
public:
    explicit BsplineCurve(const std::vector<Vector3f> &points);

    ~BsplineCurve();

    void discretize(int resolution, std::vector<CurvePoint> &data) override;

    CurvePoint curve_point_at_t(double t) override;

protected:
    int n;
    int k = 3;
    std::vector<double> knots;

private:
    double **B;
};
