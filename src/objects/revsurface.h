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

    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> real_0_1_distribution;
    static std::uniform_real_distribution<double> real_10_distribution;

public:
    RevSurface(Curve *pCurve, Material *material);

    ~RevSurface();

    bool intersect(const Ray &ray, Hit &hit, float t_min) override;
};

#endif // REVSURFACE_HPP
