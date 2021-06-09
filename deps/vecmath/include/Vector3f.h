#ifndef VECTOR_3F_H
#define VECTOR_3F_H

class Vector2f;

class Vector3f {
public:
    static const Vector3f ZERO;
    static const Vector3f UP;
    static const Vector3f RIGHT;
    static const Vector3f FORWARD;

    __host__ __device__ Vector3f(float f = 0.f);

    __host__ __device__ Vector3f(float x, float y, float z);

    __host__ __device__ Vector3f(const Vector2f &xy, float z);

    __host__ __device__ Vector3f(float x, const Vector2f &yz);

    // copy constructors
    __host__ __device__ Vector3f(const Vector3f &rv);

    // assignment operators
    __host__ __device__ Vector3f &operator=(const Vector3f &rv);

    // no destructor necessary

    // returns the ith element
    __host__ __device__ const float &operator[](int i) const;

    __host__ __device__ float &operator[](int i);

    __host__ __device__ float &x();

    __host__ __device__ float &y();

    __host__ __device__ float &z();

    __host__ __device__ float x() const;

    __host__ __device__ float y() const;

    __host__ __device__ float z() const;

    __host__ __device__ Vector2f xy() const;

    __host__ __device__ Vector2f xz() const;

    __host__ __device__ Vector2f yz() const;

    __host__ __device__ Vector3f xyz() const;

    __host__ __device__ Vector3f yzx() const;

    __host__ __device__ Vector3f zxy() const;

    __host__ __device__ float length() const;

    __host__ __device__ float squaredLength() const;

    __host__ __device__ float normalize();

    __host__ __device__ Vector3f normalized() const;

    __host__ __device__ Vector2f homogenized() const;

    __host__ __device__ void negate();

    // ---- Utility ----
    operator const float *() const; // automatic type conversion for OpenGL
    operator float *();             // automatic type conversion for OpenGL
    void print() const;

    __host__ __device__ Vector3f &operator+=(const Vector3f &v);

    __host__ __device__ Vector3f &operator-=(const Vector3f &v);

    __host__ __device__ Vector3f &operator*=(float f);

    __host__ __device__ Vector3f &operator*=(const Vector3f &v);

    __host__ __device__ static float dot(const Vector3f &v0, const Vector3f &v1);

    __host__ __device__ static Vector3f cross(const Vector3f &v0, const Vector3f &v1);

    // computes the linear interpolation between v0 and v1 by alpha \in [0,1]
    // returns v0 * ( 1 - alpha ) * v1 * alpha
    __host__ __device__ static Vector3f lerp(const Vector3f &v0, const Vector3f &v1, float alpha);

    // computes the cubic catmull-rom interpolation between p0, p1, p2, p3
    // by t \in [0,1].  Guarantees that at t = 0, the result is p0 and
    // at p1, the result is p2.
    __host__ __device__ static Vector3f cubicInterpolate(const Vector3f &p0, const Vector3f &p1,
                                                         const Vector3f &p2, const Vector3f &p3,
                                                         float t);

private:
    float m_elements[3];
};

// component-wise operators
__host__ __device__ Vector3f operator+(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ Vector3f operator-(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ Vector3f operator*(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ Vector3f operator/(const Vector3f &v0, const Vector3f &v1);

// unary negation
__host__ __device__ Vector3f operator-(const Vector3f &v);

// multiply and divide by scalar
__host__ __device__ Vector3f operator*(float f, const Vector3f &v);

__host__ __device__ Vector3f operator*(const Vector3f &v, float f);

__host__ __device__ Vector3f operator/(const Vector3f &v, float f);

__host__ __device__ bool operator==(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ bool operator!=(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ bool operator<(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ bool operator>(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ bool operator<=(const Vector3f &v0, const Vector3f &v1);

__host__ __device__ bool operator>=(const Vector3f &v0, const Vector3f &v1);

#endif // VECTOR_3F_H
