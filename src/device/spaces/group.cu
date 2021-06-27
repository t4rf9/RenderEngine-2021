#include "group.h"

__device__ Group::Group(int capacity) : capacity(capacity), num_objects(0) {
    objects = new Object3D *[capacity];
}

__device__ Group::Group(int num_objects, Object3D **objects)
    : num_objects(num_objects), objects(objects), capacity(num_objects) {}

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

__device__ bool Group::intersect(const Ray &ray, float t_min, float t_max,
                                 RandState &rand_state) {
    for (int i = 0; i < num_objects; i++) {
        if (objects[i] != nullptr &&
            objects[i]->intersect(ray, t_min, t_max, rand_state)) {
            return true;
        }
    }
    return false;
}

__device__ void Group::addObject(Object3D *obj) {
    if (num_objects == capacity) {
        capacity *= 2;
        Object3D **pObjects_new = new Object3D *[capacity];
        memcpy(pObjects_new, objects, num_objects * sizeof(Object3D *));
        delete[] objects;
        objects = pObjects_new;
    }
    objects[num_objects++] = obj;
}

__device__ void Group::shrink_to_fit() {
    if (num_objects < capacity) {
        capacity = num_objects;
        Object3D **pObjects_new = new Object3D *[capacity];
        memcpy(pObjects_new, objects, num_objects * sizeof(Object3D *));
        delete[] objects;
        objects = pObjects_new;
    }
}
