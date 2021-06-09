#pragma once

#include <vecmath.h>

#include "ray.h"

class Material;

class Hit {
public:
    // constructors
    Hit();

    Hit(float t, Material *m, const Vector3f &n);

    Hit(const Hit &h) = default;

    // destructor
    ~Hit() = default;

    inline float getT() const { return t; }

    inline void clear() { t = 1e38; }

    inline Material *getMaterial() const { return material; }

    inline const Vector3f &getNormal() const { return normal; }

    inline Vector3f &getNormal_var() { return normal; }

    inline void set(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
    }

private:
    float t;
    Material *material;
    Vector3f normal;
};

inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
    auto n = h.getNormal();
    os << "Hit <" << h.getT() << ", (" << n[0] << ", " << n[1] << ", " << n[2] << ")>";
    return os;
}
