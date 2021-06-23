#pragma once

#include <vecmath.h>

#include "object3d.h"

// transforms a 3D point using a matrix, returning a 3D point
// Ao1 + b = o
__device__ static Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point);

// transform a 3D direction using a matrix, returning a direction
// Ad1 = d
__device__ static Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir);

class Transform : public Object3D {
public:
    __device__ Transform() = delete;

    /**
     * @param m [A, b; 0, 1], m[x; 1] = [Ax + b; 1], m[x; 0] = [Ax; 0]
     * @param obj
     */
    __device__ Transform(const Matrix4f &m, Object3D *obj);

    __device__ ~Transform() override;

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              RandState &rand_state) override;

    __device__ bool intersect(const Ray &ray, float t_min, float t_max,
                              RandState &rand_state) override;

    __device__ inline Object3D *getObject() const { return o; }

protected:
    Object3D *o;        // un-transformed object
    Matrix4f transform; // inverted
};
