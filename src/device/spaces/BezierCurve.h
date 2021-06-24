#pragma once

#include "curve.h"

class BezierCurve : public Curve {
public:
    __device__ explicit BezierCurve(Vector3f *points, int num_controls);

    __device__ ~BezierCurve();

    __device__ int discretize(int resolution, Vector3f **points) override;

    __device__ CurvePoint curve_point_at_t(float t) override;

    __device__ Vector3f point_at_t(float t) override;

private:
    int n;
};
