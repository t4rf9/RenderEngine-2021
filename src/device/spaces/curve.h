#ifndef CURVE_HPP
#define CURVE_HPP

#include <vecmath.h>

#include "BoundingBox.h"
#include "object3d.h"

// PA3: Implement Bernstein class to compute spline basis function.
//       You may refer to the python-script for implementation.

// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in
// all the data.
struct CurvePoint {
    Vector3f V; // Vertex
    Vector3f T; // Tangent  (unit)
};

class Curve : public Object3D {
protected:
    Vector3f *controls;
    int num_controls;

    BoundingBox *pBox;

public:
    __device__ explicit Curve(Vector3f *points, int num_controls);

    __device__ virtual ~Curve();

    __device__ virtual bool intersect(const Ray &ray, Hit &hit, float t_min,
                                      RandState &rand_state) override;

    __device__ virtual bool intersect(const Ray &ray, float t_min, float t_max,
                                      RandState &rand_state) override;

    __device__ bool IsFlat() const;

    __device__ virtual int discretize(int resolution, Vector3f **points) = 0;

    __device__ virtual CurvePoint curve_point_at_t(float t) = 0;

    __device__ virtual Vector3f point_at_t(float t) = 0;

    __device__ inline BoundingBox *get_bounding_box() { return pBox; }
};

#endif // CURVE_HPP
