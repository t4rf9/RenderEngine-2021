#ifndef MATRIX3F_H
#define MATRIX3F_H

#include <cstdio>

class Matrix2f;

class Quat4f;

class Vector3f;

// 3x3 Matrix, stored in column major order (OpenGL style)
class Matrix3f {
public:
    // Fill a 3x3 matrix with "fill", default to 0.
    __host__ __device__ Matrix3f(float fill = 0.f);

    __host__ __device__ Matrix3f(float m00, float m01, float m02, float m10, float m11, float m12,
                                 float m20, float m21, float m22);

    // setColumns = true ==> sets the columns of the matrix to be [v0 v1 v2]
    // otherwise, sets the rows
    __host__ __device__ Matrix3f(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2,
                                 bool setColumns = true);

    __host__ __device__ Matrix3f(const Matrix3f &rm);            // copy constructor
    __host__ __device__ Matrix3f &operator=(const Matrix3f &rm); // assignment operator
    // no destructor necessary

    __host__ __device__ const float &operator()(int i, int j) const;

    __host__ __device__ float &operator()(int i, int j);

    __host__ __device__ Vector3f getRow(int i) const;

    __host__ __device__ void setRow(int i, const Vector3f &v);

    __host__ __device__ Vector3f getCol(int j) const;

    __host__ __device__ void setCol(int j, const Vector3f &v);

    // gets the 2x2 submatrix of this matrix to m
    // starting with upper left corner at (i0, j0)
    __host__ __device__ Matrix2f getSubmatrix2x2(int i0, int j0) const;

    // sets a 2x2 submatrix of this matrix to m
    // starting with upper left corner at (i0, j0)
    __host__ __device__ void setSubmatrix2x2(int i0, int j0, const Matrix2f &m);

    __host__ __device__ float determinant() const;

    __host__ __device__ Matrix3f
    inverse(bool *pbIsSingular = NULL,
            float epsilon = 0.f) const; // TODO: invert in place as well

    __host__ __device__ void transpose();

    __host__ __device__ Matrix3f transposed() const;

    // ---- Utility ----
    operator float *(); // automatic type conversion for GL
    void print();

    __host__ __device__ static float determinant3x3(float m00, float m01, float m02, float m10,
                                                    float m11, float m12, float m20, float m21,
                                                    float m22);

    __host__ __device__ static Matrix3f ones();

    __host__ __device__ static Matrix3f identity();

    __host__ __device__ static Matrix3f rotateX(float radians);

    __host__ __device__ static Matrix3f rotateY(float radians);

    __host__ __device__ static Matrix3f rotateZ(float radians);

    __host__ __device__ static Matrix3f scaling(float sx, float sy, float sz);

    __host__ __device__ static Matrix3f uniformScaling(float s);

    __host__ __device__ static Matrix3f rotation(const Vector3f &rDirection, float radians);

    // Returns the rotation matrix represented by a unit quaternion
    // if q is not normalized, it it normalized first
    __host__ __device__ static Matrix3f rotation(const Quat4f &rq);

private:
    float m_elements[9];
};

// Matrix-Vector multiplication
// 3x3 * 3x1 ==> 3x1
__host__ __device__ Vector3f operator*(const Matrix3f &m, const Vector3f &v);

// Matrix-Matrix multiplication
__host__ __device__ Matrix3f operator*(const Matrix3f &x, const Matrix3f &y);

#endif // MATRIX3F_H
