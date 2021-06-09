#ifndef QUAT4F_H
#define QUAT4F_H

class Vector3f;

class Vector4f;

#include "Matrix3f.h"

class Quat4f {
public:
    static const Quat4f ZERO;
    static const Quat4f IDENTITY;

    __host__ __device__ Quat4f();

    // q = w + x * i + y * j + z * k
    __host__ __device__ Quat4f(float w, float x, float y, float z);

    __host__ __device__ Quat4f(const Quat4f &rq);            // copy constructor
    __host__ __device__ Quat4f &operator=(const Quat4f &rq); // assignment operator
    // no destructor necessary

    // returns a quaternion with 0 real part
    __host__ __device__ Quat4f(const Vector3f &v);

    // copies the components of a Vector4f directly into this quaternion
    __host__ __device__ Quat4f(const Vector4f &v);

    // returns the ith element
    __host__ __device__ const float &operator[](int i) const;

    __host__ __device__ float &operator[](int i);

    __host__ __device__ float w() const;

    __host__ __device__ float x() const;

    __host__ __device__ float y() const;

    __host__ __device__ float z() const;

    __host__ __device__ Vector3f xyz() const;

    __host__ __device__ Vector4f wxyz() const;

    __host__ __device__ float abs() const;

    __host__ __device__ float absSquared() const;

    __host__ __device__ void normalize();

    __host__ __device__ Quat4f normalized() const;

    __host__ __device__ void conjugate();

    __host__ __device__ Quat4f conjugated() const;

    __host__ __device__ void invert();

    __host__ __device__ Quat4f inverse() const;

    // log and exponential maps
    __host__ __device__ Quat4f log() const;

    __host__ __device__ Quat4f exp() const;

    // returns unit vector for rotation and radians about the unit vector
    __host__ __device__ Vector3f getAxisAngle(float *radiansOut);

    // sets this quaternion to be a rotation of fRadians about v = < fx, fy, fz >,
    // v need not necessarily be unit length
    __host__ __device__ void setAxisAngle(float radians, const Vector3f &axis);

    // ---- Utility ----
    void print();

    // quaternion dot product (a la vector)
    __host__ __device__ static float dot(const Quat4f &q0, const Quat4f &q1);

    // linear (stupid) interpolation
    __host__ __device__ static Quat4f lerp(const Quat4f &q0, const Quat4f &q1, float alpha);

    // spherical linear interpolation
    __host__ __device__ static Quat4f slerp(const Quat4f &a, const Quat4f &b, float t,
                                            bool allowFlip = true);

    // spherical quadratic interoplation between a and b at point t
    // given quaternion tangents tanA and tanB (can be computed using
    // squadTangent)
    __host__ __device__ static Quat4f squad(const Quat4f &a, const Quat4f &tanA, const Quat4f &tanB,
                                            const Quat4f &b, float t);

    __host__ __device__ static Quat4f cubicInterpolate(const Quat4f &q0, const Quat4f &q1,
                                                       const Quat4f &q2, const Quat4f &q3, float t);

    // Log-difference between a and b, used for squadTangent
    // returns log( a^-1 b )
    __host__ __device__ static Quat4f logDifference(const Quat4f &a, const Quat4f &b);

    // Computes a tangent at center, defined by the before and after quaternions
    // Useful for squad()
    __host__ __device__ static Quat4f squadTangent(const Quat4f &before, const Quat4f &center,
                                                   const Quat4f &after);

    __host__ __device__ static Quat4f fromRotationMatrix(const Matrix3f &m);

    __host__ __device__ static Quat4f fromRotatedBasis(const Vector3f &x, const Vector3f &y,
                                                       const Vector3f &z);

    // returns a unit quaternion that's a uniformly distributed rotation
    // given u[i] is a uniformly distributed random number in [0,1]
    // taken from Graphics Gems II
    __host__ __device__ static Quat4f randomRotation(float u0, float u1, float u2);

private:
    float m_elements[4];
};

__host__ __device__ Quat4f operator+(const Quat4f &q0, const Quat4f &q1);

__host__ __device__ Quat4f operator-(const Quat4f &q0, const Quat4f &q1);

__host__ __device__ Quat4f operator*(const Quat4f &q0, const Quat4f &q1);

__host__ __device__ Quat4f operator*(float f, const Quat4f &q);

__host__ __device__ Quat4f operator*(const Quat4f &q, float f);

#endif // QUAT4F_H
