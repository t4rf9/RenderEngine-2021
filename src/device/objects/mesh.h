#pragma once

#include <vecmath.h>

#include "BoundingBox.h"
#include "object3d.h"
#include "triangle.h"

class Mesh : public Object3D {
public:
    __device__ Mesh(Vector3f *vertices, int num_vertices, dim3 *face_indices,
                    int num_faces, Material *material);

    __device__ ~Mesh();

    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state) override;

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state) override;

private:
    int num_vertices;
    int num_faces;
    Vector3f *vertices;
    Triangle **faces;

    BoundingBox *pBox;
};
