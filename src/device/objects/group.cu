#include "group.h"

__device__ Group::Group(int num_objects) : num_objects(num_objects) {
    objects = new Object3D *[num_objects];
    memset(objects, 0, num_objects * sizeof(Object3D *));
}

__device__ Group::~Group() {
    // objects[i] are deleted by the Scene to prevent recursive function calls.
    delete[] objects;
}

__device__ bool Group::intersect(const Ray &ray, Hit &hit, float t_min,
                                 RandState &rand_state) {
    bool res = false;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i] != nullptr && objects[i]->intersect(ray, hit, t_min, rand_state)) {
            res = true;
        }
    }
    return res;
}

__device__ void Group::addObject(int index, Object3D *obj) { objects[index] = obj; }
