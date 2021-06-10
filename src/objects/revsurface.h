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

public:
    RevSurface(Curve *pCurve, Material *material);

    ~RevSurface();

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState *rand_state) override;
};

#endif // REVSURFACE_HPP
