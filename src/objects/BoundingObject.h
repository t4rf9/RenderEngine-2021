#pragma once

#include "hit.h"
#include "ray.h"

class BoundingObject {
public:
    virtual bool intersect(const Ray &ray, float t_min) = 0;

private:
};
