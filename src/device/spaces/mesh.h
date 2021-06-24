#pragma once

#include <vecmath.h>

#include "BoundingBox.h"
#include "define.h"
#include "object3d.h"
#include "triangle.h"

class Mesh : public Object3D {
public:
    __device__ Mesh(Vector3f *triangle_vertices, int num_triangles, Vector3f min,
                    Vector3f max, Material *material, float curve_step = 0.f);

    __device__ ~Mesh();

    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state) override;

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state) override;

    __device__ bool intersect_rev(const Ray &ray, Hit &hit, float t_min,
                                  RandState &rand_state);

    __device__ bool intersect_rev(const Ray &ray, float t_min, float t_max,
                                  RandState &rand_state);

private:
    float curve_step;
    int num_triangles;
    Triangle **triangles;

    BoundingBox *pBox;
};
