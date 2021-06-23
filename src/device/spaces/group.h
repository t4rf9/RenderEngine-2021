#pragma once

#include "device/hit.h"
#include "device/ray.h"
#include "object3d.h"

class Group : public Object3D {
public:
    __device__ Group() = delete;

    __device__ explicit Group(int num_objects);

    __device__ ~Group() override;

    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state) override;

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state) override;

    __device__ void addObject(int index, Object3D *obj);

    __device__ inline Object3D *getObject(int index) const { return objects[index]; }

    __device__ inline int getGroupSize() { return num_objects; }

private:
    int num_objects;
    Object3D **objects;
};
