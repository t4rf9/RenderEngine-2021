#pragma once

#include <vecmath.h>

struct CurveParams {
    enum Type { Bezier, BSpline } type;
    Vector3f *controls;
    int num_controls;
};
