#include "mesh.h"

__device__ Mesh::Mesh(Vector3f *triangle_vertices, int num_triangles, Vector3f min,
                      Vector3f max, Material *material, float curve_step)
    : Object3D(material), num_triangles(num_triangles), curve_step(curve_step) {
    triangles = new Triangle *[num_triangles];
    for (int i = 0; i < num_triangles; i++) {
        triangles[i] =
            new Triangle(triangle_vertices[3 * i], triangle_vertices[3 * i + 1],
                         triangle_vertices[3 * i + 2], material);
    }

    pBox = new BoundingBox(min, max);
}

__device__ Mesh::~Mesh() {
    delete pBox;
    for (int i = 0; i < num_triangles; i++) {
        delete triangles[i];
    }
    delete[] triangles;
}

__device__ bool Mesh::intersect(const Ray &ray, Hit &hit, float t_min,
                                RandState &rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    bool result = false;
    for (int i = 0; i < num_triangles; ++i) {
        bool res_local = triangles[i]->intersect(ray, hit, t_min, rand_state);
        result = result || res_local;
    }
    return result;
}

__device__ bool Mesh::intersect(const Ray &ray, float t_min, float t_max,
                                RandState &rand_state) {
    if (!pBox->intersect(ray, t_min, t_max)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    for (int i = 0; i < num_triangles; ++i) {
        if (triangles[i]->intersect(ray, t_min, t_max, rand_state)) {
            return true;
        }
    }
    return false;
}

__device__ bool Mesh::intersect_rev(const Ray &ray, Hit &hit, float t_min,
                                    RandState &rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    bool result = false;
    for (int i = 0; i < num_triangles; ++i) {
        bool res_local = triangles[i]->intersect(ray, hit, t_min, rand_state);
        if (res_local) {
            int u_level = i / (angle_steps * 2);
            float u = float(u_level) * curve_step;

            Vector3f point = ray.pointAtParameter(hit.getT());
            float r = sqrt(point.x() * point.x() + point.z() * point.z());
            float sinv = point.z() / r;
            float cosv = point.x() / r;
            float v = sinv / (1.f + cosv);
            hit.set(hit.getT(), u, v);
        }
        result = result || res_local;
    }
    return result;
}

__device__ bool Mesh::intersect_rev(const Ray &ray, float t_min, float t_max,
                                    RandState &rand_state) {
    if (!pBox->intersect(ray, t_min, t_max)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    for (int i = 0; i < num_triangles; ++i) {
        if (triangles[i]->intersect(ray, t_min, t_max, rand_state)) {
            return true;
        }
    }
    return false;
}