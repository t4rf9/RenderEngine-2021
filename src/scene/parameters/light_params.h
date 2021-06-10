#pragma once

#include <vecmath.h>

struct LightParams {
    enum Type { Directional, Point } type;

    union {
        Vector3f position;
        Vector3f direction;
    };

    Vector3f color;
};
