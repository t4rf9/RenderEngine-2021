#pragma once

#include "space.h"

class BSPNode : public Space {
private:
    enum Axis { X = 0, Y = 1, Z = 2 } axis;
    float value;
    Space *lChild;
    Space *rChild;

public:
    __device__ BSPNode(Axis axis, float value);
    __device__ ~BSPNode();

    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state);

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state);
};
