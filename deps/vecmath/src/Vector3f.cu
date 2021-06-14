#include "Vector3f.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Vector2f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
const Vector3f Vector3f::ZERO = Vector3f(0.f, 0.f, 0.f);

// static
const Vector3f Vector3f::UP = Vector3f(0.f, 1.f, 0.f);

// static
const Vector3f Vector3f::RIGHT = Vector3f(1.f, 0.f, 0.f);

// static
const Vector3f Vector3f::FORWARD = Vector3f(0.f, 0.f, -1.f);

__host__ __device__ Vector3f::Vector3f(float f) {
    m_elements[0] = f;
    m_elements[1] = f;
    m_elements[2] = f;
}

__host__ __device__ Vector3f::Vector3f(float x, float y, float z) {
    m_elements[0] = x;
    m_elements[1] = y;
    m_elements[2] = z;
}

__host__ __device__ Vector3f::Vector3f(const Vector2f &xy, float z) {
    m_elements[0] = xy.x();
    m_elements[1] = xy.y();
    m_elements[2] = z;
}

__host__ __device__ Vector3f::Vector3f(float x, const Vector2f &yz) {
    m_elements[0] = x;
    m_elements[1] = yz.x();
    m_elements[2] = yz.y();
}

__host__ __device__ Vector3f::Vector3f(const Vector3f &rv) {
    m_elements[0] = rv[0];
    m_elements[1] = rv[1];
    m_elements[2] = rv[2];
}

__host__ __device__ Vector3f &Vector3f::operator=(const Vector3f &rv) {
    m_elements[0] = rv[0];
    m_elements[1] = rv[1];
    m_elements[2] = rv[2];
    return *this;
}

__host__ __device__ const float &Vector3f::operator[](int i) const {
    return m_elements[i];
}

__host__ __device__ float &Vector3f::operator[](int i) { return m_elements[i]; }

__host__ __device__ float &Vector3f::x() { return m_elements[0]; }

__host__ __device__ float &Vector3f::y() { return m_elements[1]; }

__host__ __device__ float &Vector3f::z() { return m_elements[2]; }

__host__ __device__ float Vector3f::x() const { return m_elements[0]; }

__host__ __device__ float Vector3f::y() const { return m_elements[1]; }

__host__ __device__ float Vector3f::z() const { return m_elements[2]; }

__host__ __device__ Vector2f Vector3f::xy() const {
    return Vector2f(m_elements[0], m_elements[1]);
}

__host__ __device__ Vector2f Vector3f::xz() const {
    return Vector2f(m_elements[0], m_elements[2]);
}

__host__ __device__ Vector2f Vector3f::yz() const {
    return Vector2f(m_elements[1], m_elements[2]);
}

__host__ __device__ Vector3f Vector3f::xyz() const {
    return Vector3f(m_elements[0], m_elements[1], m_elements[2]);
}

__host__ __device__ Vector3f Vector3f::yzx() const {
    return Vector3f(m_elements[1], m_elements[2], m_elements[0]);
}

__host__ __device__ Vector3f Vector3f::zxy() const {
    return Vector3f(m_elements[2], m_elements[0], m_elements[1]);
}

__host__ __device__ float Vector3f::length() const {
    return sqrt(m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] +
                m_elements[2] * m_elements[2]);
}

__host__ __device__ float Vector3f::squaredLength() const {
    return (m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] +
            m_elements[2] * m_elements[2]);
}

__host__ __device__ float Vector3f::normalize() {
    float norm = length();
    m_elements[0] /= norm;
    m_elements[1] /= norm;
    m_elements[2] /= norm;
    return norm;
}

__host__ __device__ Vector3f Vector3f::normalized() const {
    float norm = length();
    return Vector3f(m_elements[0] / norm, m_elements[1] / norm, m_elements[2] / norm);
}

__host__ __device__ Vector2f Vector3f::homogenized() const {
    return Vector2f(m_elements[0] / m_elements[2], m_elements[1] / m_elements[2]);
}

__host__ __device__ void Vector3f::negate() {
    m_elements[0] = -m_elements[0];
    m_elements[1] = -m_elements[1];
    m_elements[2] = -m_elements[2];
}

Vector3f::operator const float *() const { return m_elements; }

Vector3f::operator float *() { return m_elements; }

void Vector3f::print() const {
    printf("< %.4f, %.4f, %.4f >\n", m_elements[0], m_elements[1], m_elements[2]);
}

__host__ __device__ Vector3f &Vector3f::operator+=(const Vector3f &v) {
    m_elements[0] += v.m_elements[0];
    m_elements[1] += v.m_elements[1];
    m_elements[2] += v.m_elements[2];
    return *this;
}

__host__ __device__ Vector3f &Vector3f::operator-=(const Vector3f &v) {
    m_elements[0] -= v.m_elements[0];
    m_elements[1] -= v.m_elements[1];
    m_elements[2] -= v.m_elements[2];
    return *this;
}

__host__ __device__ Vector3f &Vector3f::operator*=(float f) {
    m_elements[0] *= f;
    m_elements[1] *= f;
    m_elements[2] *= f;
    return *this;
}

__host__ __device__ Vector3f &Vector3f::operator*=(const Vector3f &v) {
    m_elements[0] *= v.m_elements[0];
    m_elements[1] *= v.m_elements[1];
    m_elements[2] *= v.m_elements[2];
    return *this;
}

// static
__host__ __device__ float Vector3f::dot(const Vector3f &v0, const Vector3f &v1) {
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

// static
__host__ __device__ Vector3f Vector3f::cross(const Vector3f &v0, const Vector3f &v1) {
    return Vector3f(v0.y() * v1.z() - v0.z() * v1.y(), v0.z() * v1.x() - v0.x() * v1.z(),
                    v0.x() * v1.y() - v0.y() * v1.x());
}

// static
__host__ __device__ Vector3f Vector3f::lerp(const Vector3f &v0, const Vector3f &v1,
                                            float alpha) {
    return alpha * (v1 - v0) + v0;
}

// static
__host__ __device__ Vector3f Vector3f::cubicInterpolate(const Vector3f &p0,
                                                        const Vector3f &p1,
                                                        const Vector3f &p2,
                                                        const Vector3f &p3, float t) {
    // geometric construction:
    //            t
    //   (t+1)/2     t/2
    // t+1        t	        t-1

    // bottom level
    Vector3f p0p1 = Vector3f::lerp(p0, p1, t + 1);
    Vector3f p1p2 = Vector3f::lerp(p1, p2, t);
    Vector3f p2p3 = Vector3f::lerp(p2, p3, t - 1);

    // middle level
    Vector3f p0p1_p1p2 = Vector3f::lerp(p0p1, p1p2, 0.5f * (t + 1));
    Vector3f p1p2_p2p3 = Vector3f::lerp(p1p2, p2p3, 0.5f * t);

    // top level
    return Vector3f::lerp(p0p1_p1p2, p1p2_p2p3, t);
}

__host__ __device__ Vector3f operator+(const Vector3f &v0, const Vector3f &v1) {
    return Vector3f(v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2]);
}

__host__ __device__ Vector3f operator-(const Vector3f &v0, const Vector3f &v1) {
    return Vector3f(v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]);
}

__host__ __device__ Vector3f operator*(const Vector3f &v0, const Vector3f &v1) {
    return Vector3f(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]);
}

__host__ __device__ Vector3f operator/(const Vector3f &v0, const Vector3f &v1) {
    return Vector3f(v0[0] / v1[0], v0[1] / v1[1], v0[2] / v1[2]);
}

__host__ __device__ Vector3f operator-(const Vector3f &v) {
    return Vector3f(-v[0], -v[1], -v[2]);
}

__host__ __device__ Vector3f operator*(float f, const Vector3f &v) {
    return Vector3f(v[0] * f, v[1] * f, v[2] * f);
}

__host__ __device__ Vector3f operator*(const Vector3f &v, float f) {
    return Vector3f(v[0] * f, v[1] * f, v[2] * f);
}

__host__ __device__ Vector3f operator/(const Vector3f &v, float f) {
    return Vector3f(v[0] / f, v[1] / f, v[2] / f);
}

__host__ __device__ bool operator==(const Vector3f &v0, const Vector3f &v1) {
    return v0.x() == v1.x() && v0.y() == v1.y() && v0.z() == v1.z();
}

__host__ __device__ bool operator!=(const Vector3f &v0, const Vector3f &v1) {
    return !(v0 == v1);
}

__host__ __device__ bool operator<(const Vector3f &v0, const Vector3f &v1) {
    return v0.x() < v1.x() && v0.y() < v1.y() && v0.z() < v1.z();
}

__host__ __device__ bool operator>(const Vector3f &v0, const Vector3f &v1) {
    return v0.x() > v1.x() && v0.y() > v1.y() && v0.z() < v1.z();
}

__host__ __device__ bool operator<=(const Vector3f &v0, const Vector3f &v1) {
    return v0.x() <= v1.x() && v0.y() <= v1.y() && v0.z() <= v1.z();
}

__host__ __device__ bool operator>=(const Vector3f &v0, const Vector3f &v1) {
    return v0.x() >= v1.x() && v0.y() >= v1.y() && v0.z() >= v1.z();
}