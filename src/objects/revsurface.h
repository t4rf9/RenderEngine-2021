#ifndef REVSURFACE_HPP
#define REVSURFACE_HPP

#include <cmath>
#include <random>
#include <tuple>

#include "BoundingBox.h"
#include "BoundingObject.h"
#include "curve.h"
#include "object3d.h"

class RevSurface : public Object3D {
    Curve *pCurve;
    BoundingBox *pBound = nullptr;

    float random01(uint_fast32_t &rand);

public:
    RevSurface(Curve *pCurve, Material *material);

    ~RevSurface();

    bool intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) override;
};

#endif // REVSURFACE_HPP
