#pragma once

#include <vecmath.h>

struct MeshParams {
    Vector3f *triangle_vertices;
    Vector3f min;
    Vector3f max;

    int num_triangles;

    int material_id;
};
