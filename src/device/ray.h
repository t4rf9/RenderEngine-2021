#pragma once

#include <Vector3f.h>

#include <cassert>
#include <iostream>

// Ray class mostly copied from Peter Shirley and Keith Morley
class Ray {
public:
    __device__ Ray() = delete;

    __device__ Ray(const Vector3f &origin, const Vector3f &direction, int depth, float weight,
        float incident_refractive_index);

    __device__ Ray(const Ray &ray);

    __device__ const Vector3f &getOrigin() const { return origin; }

    __device__ const Vector3f &getDirection() const { return direction; }

    __device__ Vector3f pointAtParameter(float t) const { return origin + direction * t; }

    __device__ inline int get_depth() const { return depth; }

    __device__ inline float get_weight() const { return weight; }

    __device__ inline float get_incident_refractive_index() const { return incident_refractive_index; }

private:
    Vector3f origin;
    Vector3f direction;
    int depth;
    float weight;
    float incident_refractive_index;
};

//std::ostream &operator<<(std::ostream &out, const Ray &ray);
