#pragma once

#include <Vector3f.h>

#include "cuda_error.h"

class Light {
public:
    Light() = default;

    virtual ~Light() = default;

    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

    virtual void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const = 0;
};
