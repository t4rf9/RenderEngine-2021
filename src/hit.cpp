#include "hit.h"

Hit::Hit() : material(nullptr), t(1e38) {}

Hit::Hit(float t, Material *m, const Vector3f &n) : t(t), material(m), normal(n) {}
