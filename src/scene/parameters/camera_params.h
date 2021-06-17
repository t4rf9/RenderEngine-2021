#pragma once

#include <vecmath.h>

struct CameraParams {
    enum Type { PerspectiveCamera } type;

    Vector3f pos;
    Vector3f direction;
    Vector3f up;

    int width;
    int height;
    float angle;

    float focus_dist;
    float aperture;
};
