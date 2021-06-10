#include "group.h"

Group::Group(int num_objects) : num(num_objects) {
    checkCudaErrors(cudaMallocManaged(&objects, num * sizeof(Object3D *)));
    memset(objects, 0, num * sizeof(Object3D *));
}

Group::~Group() { checkCudaErrors(cudaFree(objects)); }

__device__ bool Group::intersect(const Ray &ray, Hit &hit, float t_min, curandState *rand_state) {
    bool res = false;
    for (int i = 0; i < num; i++) {
        if (objects[i] != nullptr && objects[i]->intersect(ray, hit, t_min, rand_state)) {
            res = true;
        }
    }
    return res;
}

void Group::addObject(int index, Object3D *obj) { objects[index] = obj; }
