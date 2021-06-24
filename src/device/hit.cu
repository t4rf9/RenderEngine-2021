#include "hit.h"

__device__ Hit::Hit() : material(nullptr), t(INFINITY) {}

__device__ Hit::Hit(float t, Material *m, const Vector3f &n)
    : t(t), material(m), normal(n) {}

__device__ Hit::Hit(float t, const Vector3f &color)
    : t(t), material(nullptr), color(color) {}

__device__ Hit::Hit(const Hit &h) : t(h.t), material(h.material), normal(h.normal) {}

__device__ Hit::~Hit() {}
