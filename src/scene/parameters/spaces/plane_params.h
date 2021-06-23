#pragma once

#include <vecmath.h>

struct PlaneParams {
    Vector3f normal;
    float d;
    Vector3f texture_origin;
    Vector3f texture_x;
    Vector3f texture_y;
    int material_id;
};
