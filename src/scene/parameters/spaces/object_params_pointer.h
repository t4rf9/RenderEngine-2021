#pragma once

struct TriangleParams;
struct SphereParams;
struct PlaneParams;
struct MeshParams;
struct RevsurfaceParams;
struct TransformParams;
struct GroupParams;
struct CurveParams;

union ObjectParamsPointer {
    TriangleParams *triangle;
    SphereParams *sphere;
    PlaneParams *plane;
    MeshParams *mesh;
    RevsurfaceParams *revsurface;
    TransformParams *transform;
    GroupParams *group;
    CurveParams *curve;
};
