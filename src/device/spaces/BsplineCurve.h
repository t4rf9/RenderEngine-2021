#pragma once

#include "curve.h"

class BsplineCurve : public Curve {
public:
    __device__ explicit BsplineCurve(Vector3f *points, int num_controls);

    __device__ ~BsplineCurve();

    __device__ int discretize(int resolution, Vector3f **points) override;

    __device__ CurvePoint curve_point_at_t(float t) override;

    __device__ Vector3f point_at_t(float t) override;

protected:
    int n;
    int k = 3;
    float *knots;

private:
};
