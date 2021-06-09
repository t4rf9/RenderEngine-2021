#pragma once

#include "Vector2f.h"
#include "Vector3f.h"
#include "objects/BoundingObject.h"

class BoundingBox : public BoundingObject {
public:
    BoundingBox() = delete;

    BoundingBox(const Vector3f &min, const Vector3f &max);

    bool intersect(const Ray &ray, float t_min) override;

    inline const Vector3f &get_min() { return min; }

    inline const Vector3f &get_max() { return max; }

private:
    Vector3f min;
    Vector3f max;
};