#pragma once

#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_error.h"
#include "hit.h"
#include "object3d.h"
#include "ray.h"

class Group : public Object3D {
public:
    Group() = delete;

    explicit Group(int num_objects);

    ~Group() override;

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState *rand_state) override;

    void addObject(int index, Object3D *obj);

    inline int getGroupSize() { return num; }

private:
    int num;
    Object3D **objects;
};
