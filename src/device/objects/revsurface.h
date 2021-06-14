#ifndef REVSURFACE_HPP
#define REVSURFACE_HPP

#include "BoundingBox.h"
#include "BoundingObject.h"
#include "curve.h"
#include "object3d.h"

class RevSurface : public Object3D {
    Curve *pCurve;
    BoundingBox *pBound = nullptr;

public:
    __device__ RevSurface(Curve *pCurve, Material *material);

    __device__ ~RevSurface();

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState &rand_state) override;
};

#endif // REVSURFACE_HPP
