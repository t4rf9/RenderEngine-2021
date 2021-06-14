#include "group.h"

Group::Group(int num_objects) : num(num_objects), objects(new Object3D *[num]) {
    memset(objects, 0, num * sizeof(Object3D *));
}

Group::~Group() { delete[] objects; }

bool Group::intersect(const Ray &ray, Hit &hit, float t_min, uint_fast32_t &rand) {
    bool res = false;
    for (int i = 0; i < num; i++) {
        if (objects[i] != nullptr && objects[i]->intersect(ray, hit, t_min, rand)) {
            res = true;
        }
    }
    return res;
}

void Group::addObject(int index, Object3D *obj) { objects[index] = obj; }