#include "ray.h"

__device__ Ray::Ray() : depth(0), weight(0), incident_refractive_index(1.f) {}

__device__ Ray::Ray(const Vector3f &origin, const Vector3f &direction, int depth,
                    float weight, float incident_refractive_index)
    : origin(origin), direction(direction), depth(depth), weight(weight),
      incident_refractive_index(incident_refractive_index) {}

__device__ Ray::Ray(const Ray &ray)
    : origin(ray.origin), direction(ray.direction), depth(ray.depth), weight(ray.weight),
      incident_refractive_index(ray.incident_refractive_index) {}

__device__ void Ray::set(const Vector3f &origin, const Vector3f &direction, int depth,
                         float weight, float incident_refractive_index) {
    this->origin = origin;
    this->direction = direction;
    this->depth = depth;
    this->weight = weight;
    this->incident_refractive_index = incident_refractive_index;
}

/*
std::ostream &operator<<(std::ostream &out, const Ray &ray) {
    auto origin = ray.getOrigin();
    auto dir = ray.getDirection();
    out << "Ray <(" << origin[0] << ", " << origin[1] << ", " << origin[2] << "), (" <<
dir[0]
        << ", " << dir[1] << ", " << dir[2] << ")>";
    return out;
}
*/
