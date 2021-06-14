#include "triangle.h"

__device__ Triangle::Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *m)
    : Object3D(m) {
    vertices[0] = a;
    vertices[1] = b;
    vertices[2] = c;
    E1 = vertices[0] - vertices[1];
    E2 = vertices[0] - vertices[2];
    normal = Vector3f::cross(E1, E2);
    normal.normalize();
}

__device__ bool Triangle::intersect(const Ray &ray, Hit &hit, float t_min,
                                    curandState &rand_state) {
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

    return true;
}
