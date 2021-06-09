#pragma once

#include <Vector3f.h>
#include <cassert>
#include <iostream>

// Ray class mostly copied from Peter Shirley and Keith Morley
class Ray {
public:
    Ray() = delete;

    Ray(const Vector3f &origin, const Vector3f &direction, int depth, float weight,
        float incident_refractive_index);

    Ray(const Ray &ray);

    const Vector3f &getOrigin() const { return origin; }

    const Vector3f &getDirection() const { return direction; }

    Vector3f pointAtParameter(float t) const { return origin + direction * t; }

    inline int get_depth() const { return depth; }

    inline float get_weight() const { return weight; }

    inline float get_incident_refractive_index() const { return incident_refractive_index; }

private:
    Vector3f origin;
    Vector3f direction;
    int depth;
    float weight;
    float incident_refractive_index;
};

std::ostream &operator<<(std::ostream &out, const Ray &ray);
