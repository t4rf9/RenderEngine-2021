#pragma once

#include <vecmath.h>

#include "ray.h"

class Material;

class Hit {
public:
    // constructors
    __device__ Hit();

    __device__ Hit(float t, Material *m, const Vector3f &n);

    __device__ Hit(float t, const Vector3f &color);

    __device__ Hit(const Hit &h);

    // destructor
    __device__ ~Hit();

    __device__ inline float getT() const { return t; }

    __device__ inline void clear() { t = INFINITY; }

    __device__ inline Material *getMaterial() const { return material; }

    __device__ inline const Vector3f &getNormal() const { return normal; }

    __device__ inline const Vector3f &getColor() const { return color; }

    __device__ inline const Vector3f &getRevParams() const { return rev_params; }

    __device__ inline Vector3f &getNormal_var() { return normal; }

    __device__ inline void set(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
    }

    __device__ inline void set(float t, float u, float v) {
        rev_params[0] = t;
        rev_params[1] = u;
        rev_params[2] = v;
        material = nullptr;
    }

private:
    float t;
    Material *material;
    union {
        Vector3f normal;
        Vector3f color;
        Vector3f rev_params;
    };
};
/*
inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
    auto n = h.getNormal();
    os << "Hit <" << h.getT() << ", (" << n[0] << ", " << n[1] << ", " << n[2] << ")>";
    return os;
}
*/
