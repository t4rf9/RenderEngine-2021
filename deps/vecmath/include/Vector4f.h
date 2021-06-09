#ifndef VECTOR_4F_H
#define VECTOR_4F_H

class Vector2f;

class Vector3f;

class Vector4f {
public:
    __host__ __device__ Vector4f(float f = 0.f);

    __host__ __device__ Vector4f(float fx, float fy, float fz, float fw);

    __host__ __device__ Vector4f(float buffer[4]);

    __host__ __device__ Vector4f(const Vector2f &xy, float z, float w);

    __host__ __device__ Vector4f(float x, const Vector2f &yz, float w);

    __host__ __device__ Vector4f(float x, float y, const Vector2f &zw);

    __host__ __device__ Vector4f(const Vector2f &xy, const Vector2f &zw);

    __host__ __device__ Vector4f(const Vector3f &xyz, float w);

    __host__ __device__ Vector4f(float x, const Vector3f &yzw);

    // copy constructors
    __host__ __device__ Vector4f(const Vector4f &rv);

    // assignment operators
    __host__ __device__ Vector4f &operator=(const Vector4f &rv);

    // no destructor necessary

    // returns the ith element
    __host__ __device__ const float &operator[](int i) const;

    __host__ __device__ float &operator[](int i);

    __host__ __device__ float &x();

    __host__ __device__ float &y();

    __host__ __device__ float &z();

    __host__ __device__ float &w();

    __host__ __device__ float x() const;

    __host__ __device__ float y() const;

    __host__ __device__ float z() const;

    __host__ __device__ float w() const;

    __host__ __device__ Vector2f xy() const;

    __host__ __device__ Vector2f yz() const;

    __host__ __device__ Vector2f zw() const;

    __host__ __device__ Vector2f wx() const;

    __host__ __device__ Vector3f xyz() const;

    __host__ __device__ Vector3f yzw() const;

    __host__ __device__ Vector3f zwx() const;

    __host__ __device__ Vector3f wxy() const;

    __host__ __device__ Vector3f xyw() const;

    __host__ __device__ Vector3f yzx() const;

    __host__ __device__ Vector3f zwy() const;

    __host__ __device__ Vector3f wxz() const;

    __host__ __device__ float abs() const;

    __host__ __device__ float absSquared() const;

    __host__ __device__ void normalize();

    __host__ __device__ Vector4f normalized() const;

    // if v.z != 0, v = v / v.w
    __host__ __device__ void homogenize();

    __host__ __device__ Vector4f homogenized() const;

    __host__ __device__ void negate();

    // ---- Utility ----
    operator const float *() const; // automatic type conversion for OpenGL
    operator float *();             // automatic type conversion for OpenG
    void print() const;

    __host__ __device__ static float dot(const Vector4f &v0, const Vector4f &v1);

    __host__ __device__ static Vector4f lerp(const Vector4f &v0, const Vector4f &v1, float alpha);

private:
    float m_elements[4];
};

// component-wise operators
__host__ __device__ Vector4f operator+(const Vector4f &v0, const Vector4f &v1);

__host__ __device__ Vector4f operator-(const Vector4f &v0, const Vector4f &v1);

__host__ __device__ Vector4f operator*(const Vector4f &v0, const Vector4f &v1);

__host__ __device__ Vector4f operator/(const Vector4f &v0, const Vector4f &v1);

// unary negation
__host__ __device__ Vector4f operator-(const Vector4f &v);

// multiply and divide by scalar
__host__ __device__ Vector4f operator*(float f, const Vector4f &v);

__host__ __device__ Vector4f operator*(const Vector4f &v, float f);

__host__ __device__ Vector4f operator/(const Vector4f &v, float f);

__host__ __device__ bool operator==(const Vector4f &v0, const Vector4f &v1);

__host__ __device__ bool operator!=(const Vector4f &v0, const Vector4f &v1);

#endif // VECTOR_4F_H
