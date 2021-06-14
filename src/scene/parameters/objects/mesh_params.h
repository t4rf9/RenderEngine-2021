#pragma once

struct MeshParams {
    Vector3f *vertices;
    dim3 *face_indices;

    int num_vertices;
    int num_faces;

    int material_id;
};
