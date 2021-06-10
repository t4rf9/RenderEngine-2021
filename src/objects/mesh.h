#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "BoundingBox.h"
#include "Vector2f.h"
#include "Vector3f.h"
#include "cuda_error.h"
#include "object3d.h"
#include "triangle.h"

class Mesh : public Object3D {
public:
    Mesh(const char *filename, Material *m);

    ~Mesh();

    __device__ bool intersect(const Ray &ray, Hit &hit, float t_min,
                              curandState *rand_state) override;

private:
    int num_vertices;
    int num_faces;
    Vector3f *vertices;
    Triangle **faces;

    BoundingBox *pBox;
};
