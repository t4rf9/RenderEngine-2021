#include "object3d.h"

__device__ Object3D::Object3D() : material(nullptr) {}

__device__ Object3D::Object3D(Material *material) : material(material) {}
