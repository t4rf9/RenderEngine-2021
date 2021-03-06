#include "hit.h"

__device__ Hit::Hit() : material(nullptr), t(1e38f) {}

__device__ Hit::Hit(float t, Material *m, const Vector3f &n) : t(t), material(m), normal(n) {}

__device__ Hit::Hit(const Hit &h) : t(h.t), material(h.material), normal(h.normal) {}

__device__ Hit::~Hit() {}
