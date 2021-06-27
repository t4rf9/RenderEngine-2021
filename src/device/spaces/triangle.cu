#include "triangle.h"

__device__ Triangle::Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                              Material *m, int id)
    : Object3D(m), center((a + b + c) / 3.f), id(id) {
    vertices[0] = a;
    vertices[1] = b;
    vertices[2] = c;
    E1 = vertices[0] - vertices[1];
    E2 = vertices[0] - vertices[2];
    normal = Vector3f::cross(E1, E2);
    normal.normalize();

    min = vertices[0];
    max = vertices[0];

    if (vertices[1].x() < min.x()) {
        min.x() = vertices[1].x();
    }
    if (vertices[1].y() < min.y()) {
        min.y() = vertices[1].y();
    }
    if (vertices[1].z() < min.z()) {
        min.z() = vertices[1].z();
    }
    if (vertices[1].x() > max.x()) {
        max.x() = vertices[1].x();
    }
    if (vertices[1].y() > max.y()) {
        max.y() = vertices[1].y();
    }
    if (vertices[1].z() > max.z()) {
        max.z() = vertices[1].z();
    }

    if (vertices[2].x() < min.x()) {
        min.x() = vertices[2].x();
    }
    if (vertices[2].y() < min.y()) {
        min.y() = vertices[2].y();
    }
    if (vertices[2].z() < min.z()) {
        min.z() = vertices[2].z();
    }
    if (vertices[2].x() > max.x()) {
        max.x() = vertices[2].x();
    }
    if (vertices[2].y() > max.y()) {
        max.y() = vertices[2].y();
    }
    if (vertices[2].z() > max.z()) {
        max.z() = vertices[2].z();
    }
}

__device__ bool Triangle::intersect(const Ray &ray, Hit &hit, float t_min,
                                    RandState &rand_state) {
    Vector3f S = vertices[0] - ray.getOrigin();
    const Vector3f &Rd = ray.getDirection();

    Matrix3f divisor_mat(Rd, E1, E2);
    float divisor = divisor_mat.determinant();

    Matrix3f mat1(S, E1, E2);
    Matrix3f mat2(Rd, S, E2);
    Matrix3f mat3(Rd, E1, S);

    float t = mat1.determinant() / divisor;
    if (t <= t_min || t > hit.getT()) {
        return false;
    }

    float b = mat2.determinant() / divisor;
    if (b < 0.f || b > 1.f) {
        return false;
    }

    float c = mat3.determinant() / divisor;
    if (c < 0.f || c > 1.f || b + c > 1.f) {
        return false;
    }

    hit.set(t, material, normal);
    hit.id = id;

    return true;
}

__device__ bool Triangle::intersect(const Ray &ray, float t_min, float t_max,
                                    RandState &rand_state) {
    Vector3f S = vertices[0] - ray.getOrigin();
    const Vector3f &Rd = ray.getDirection();

    Matrix3f divisor_mat(Rd, E1, E2);
    float divisor = divisor_mat.determinant();

    Matrix3f mat1(S, E1, E2);
    Matrix3f mat2(Rd, S, E2);
    Matrix3f mat3(Rd, E1, S);

    float t = mat1.determinant() / divisor;
    float b = mat2.determinant() / divisor;
    float c = mat3.determinant() / divisor;
    if (t <= t_min || t >= t_max) {
        return false;
    }

    if (b < 0.f || b > 1.f) {
        return false;
    }

    if (c < 0.f || c > 1.f || b + c > 1.f) {
        return false;
    }

    return t_min < t && t < t_max && 0.f <= b && b <= 1.f && 0.f <= c && c <= 1.f &&
           b + c <= 1.f;
}
