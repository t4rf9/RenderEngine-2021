#include "mesh.h"

__device__ bool Mesh::intersect(const Ray &ray, Hit &hit, float t_min,
                                curandState &rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    bool result = false;
    for (int i = 0; i < num_faces; ++i) {
        bool res_local = faces[i]->intersect(ray, hit, t_min, rand_state);
        result = result || res_local;
    }
    return result;
}

__device__ Mesh::Mesh(Vector3f *vertices, int num_vertices, dim3 *face_indices,
                      int num_faces, Material *material)
    : Object3D(material), vertices(vertices), num_vertices(num_vertices),
      num_faces(num_faces) {

    faces = new Triangle *[num_faces];
    for (int i = 0; i < num_faces; i++) {
        faces[i] = new Triangle(vertices[face_indices[i].x], vertices[face_indices[i].y],
                                vertices[face_indices[i].z], material);
    }

    Vector3f min = vertices[0];
    Vector3f max = vertices[0];

    for (int i = 1; i < num_vertices; i++) {
        auto &p = vertices[i];
        for (int j = 0; j < 3; j++) {
            if (p[j] < min[j]) {
                min[j] = p[j];
            }
            if (p[j] > max[j]) {
                max[j] = p[j];
            }
        }
    }

    pBox = new BoundingBox(min, max);
}

__device__ Mesh::~Mesh() {
    delete pBox;
    for (int i = 0; i < num_faces; i++) {
        delete faces[i];
    }
    delete[] faces;
}
