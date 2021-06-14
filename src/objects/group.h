#pragma once

#include <cstring>
#include <iostream>
#include <vector>

#include "hit.h"
#include "object3d.h"
#include "ray.h"

class Group : public Object3D {
public:
    Group() = delete;

    explicit Group(int num_objects);

    ~Group() override;

    bool intersect(const Ray &r, Hit &h, float tmin, uint_fast32_t &rand) override;

    void addObject(int index, Object3D *obj);

    inline int getGroupSize() { return num; }

private:
    int num;
    Object3D **objects;
};
