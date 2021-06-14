#include "object3d.h"

Object3D::Object3D() : material(nullptr) {}

Object3D::Object3D(Material *material) : material(material) {}
