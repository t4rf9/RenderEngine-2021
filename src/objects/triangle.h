#pragma once

#include <vecmath.h>

#include <cmath>
#include <iostream>

#include "object3d.h"

using namespace std;

class Triangle : public Object3D {
public:
    Triangle() = delete;

    // a b c are three vertex positions of the triangle
    Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *m);

    bool intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) override;

    inline void setNormal(const Vector3f &n) { normal = n; }

protected:
    Vector3f normal;
    Vector3f vertices[3];

    Vector3f E1;
    Vector3f E2;
};
