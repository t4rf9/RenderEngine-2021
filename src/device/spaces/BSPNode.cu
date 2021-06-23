#include "BSPNode.h"

__device__ BSPNode::BSPNode(Axis axis, float value) : Space(), axis(axis), value(value) {}

__device__ BSPNode::~BSPNode() {}

__device__ bool BSPNode::intersect(const Ray &ray, Hit &hit, float t_min,
                                   RandState &rand_state) {
    bool res = false;

    const Vector3f &o = ray.getOrigin();
    const Vector3f &d = ray.getDirection();

    Space *near;
    Space *other;
    if (o[axis] < value) {
        near = lChild;
        other = rChild;
    } else {
        near = rChild;
        other = lChild;
    }

    res = near->intersect(ray, hit, t_min, rand_state);

    float t = (value - o[axis]) / d[axis];
    if (t_min < t && t < hit.getT()) {
        res |= other->intersect(ray, hit, t, rand_state);
    }

    return res;
}

__device__ bool BSPNode::intersect(const Ray &ray, float t_min, float t_max,
                                   RandState &rand_state) {
    bool res = false;

    const Vector3f &o = ray.getOrigin();
    const Vector3f &d = ray.getDirection();

    Space *near;
    Space *other;
    if (o[axis] < value) {
        near = lChild;
        other = rChild;
    } else {
        near = rChild;
        other = lChild;
    }

    float t = (value - o[axis]) / d[axis];
    res = near->intersect(ray, t_min, t_max, rand_state);

    if (!res && t_min < t && t < t_max) {
        res = other->intersect(ray, t, t_max, rand_state);
    }

    return res;
}