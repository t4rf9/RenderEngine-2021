#pragma once

#include <vecmath.h>

struct LightParams {
    enum Type { DIRECRIONAL, POINT, DISK } type;

    union {
        Vector3f position;
        Vector3f direction;
    };

    Vector3f color;
    float radius;
    Vector3f normal;
};
