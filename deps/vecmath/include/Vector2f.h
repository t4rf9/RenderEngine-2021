#ifndef VECTOR_2F_H
#define VECTOR_2F_H

#include <cmath>

class Vector3f;

class Vector2f {
public:
    static const Vector2f ZERO;
    static const Vector2f UP;
    static const Vector2f RIGHT;

    __host__ __device__ explicit Vector2f(float f = 0.f);

    __host__ __device__ Vector2f(float x, float y);

    // copy constructors
    __host__ __device__ Vector2f(const Vector2f &rv);

    // assignment operators
    __host__ __device__ Vector2f &operator=(const Vector2f &rv);

    // no destructor necessary

    // returns the ith element
    __host__ __device__ const float &operator[](int i) const;

    __host__ __device__ float &operator[](int i);

    __host__ __device__ float &x();

    __host__ __device__ float &y();

    __host__ __device__ float x() const;

    __host__ __device__ float y() const;

    __host__ __device__ Vector2f xy() const;

    __host__ __device__ Vector2f yx() const;

    __host__ __device__ Vector2f xx() const;

    __host__ __device__ Vector2f yy() const;

    // returns ( -y, x )
    __host__ __device__ Vector2f normal() const;

    __host__ __device__ float abs() const;

    __host__ __device__ float absSquared() const;

    __host__ __device__ void normalize();

    __host__ __device__ Vector2f normalized() const;

    __host__ __device__ void negate();

    // ---- Utility ----
    operator const float *() const; // automatic type conversion for OpenGL
    operator float *();             // automatic type conversion for OpenGL
    void print() const;

    __host__ __device__ Vector2f &operator+=(const Vector2f &v);

    __host__ __device__ Vector2f &operator-=(const Vector2f &v);

    __host__ __device__ Vector2f &operator*=(float f);

    __host__ __device__ static float dot(const Vector2f &v0, const Vector2f &v1);

    __host__ __device__ static Vector3f cross(const Vector2f &v0, const Vector2f &v1);

    // returns v0 * ( 1 - alpha ) + v1 * alpha
    __host__ __device__ static Vector2f lerp(const Vector2f &v0, const Vector2f &v1,
                                             float alpha);

private:
    float m_elements[2];
};

// component-wise operators
__host__ __device__ Vector2f operator+(const Vector2f &v0, const Vector2f &v1);

__host__ __device__ Vector2f operator-(const Vector2f &v0, const Vector2f &v1);

__host__ __device__ Vector2f operator*(const Vector2f &v0, const Vector2f &v1);

__host__ __device__ Vector2f operator/(const Vector2f &v0, const Vector2f &v1);

// unary negation
__host__ __device__ Vector2f operator-(const Vector2f &v);

// multiply and divide by scalar
__host__ __device__ Vector2f operator*(float f, const Vector2f &v);

__host__ __device__ Vector2f operator*(const Vector2f &v, float f);

__host__ __device__ Vector2f operator/(const Vector2f &v, float f);

__host__ __device__ bool operator==(const Vector2f &v0, const Vector2f &v1);

__host__ __device__ bool operator!=(const Vector2f &v0, const Vector2f &v1);

#endif // VECTOR_2F_H
