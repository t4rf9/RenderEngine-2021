#pragma once

#include "space.h"

// Base class for all 3d entities.
class Object3D : public Space {
public:
    __device__ Object3D();

    __device__ virtual ~Object3D();

    __device__ explicit Object3D(Material *material);

protected:
    Material *material;
};
