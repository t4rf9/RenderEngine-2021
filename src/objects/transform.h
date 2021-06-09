#pragma once

#include <vecmath.h>

#include "object3d.h"

// transforms a 3D point using a matrix, returning a 3D point
// Ao1 + b = o
static Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point);

// transform a 3D direction using a matrix, returning a direction
// Ad1 = d
static Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir);

class Transform : public Object3D {
public:
    Transform() = delete;

    /**
     * @param m [A, b; 0, 1], m[x; 1] = [Ax + b; 1], m[x; 0] = [Ax; 0]
     * @param obj
     */
    Transform(const Matrix4f &m, Object3D *obj);

    ~Transform() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override;

protected:
    Object3D *o;        // un-transformed object
    Matrix4f transform; // inverted
};
