#pragma once

#include <cassert>
#include <vecmath.h>

#include "cuda_error.h"

class Camera;
class Light;
class Group;

class Scene {
public:
    Scene() = default;

    ~Scene();

    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

    __host__ __device__ inline Camera *getCamera() const { return camera; }

    __host__ __device__ inline Vector3f getBackgroundColor() const { return background_color; }

    __host__ __device__ inline Vector3f getEnvironmentColor() const { return environment_color; }

    __host__ __device__ inline int getNumLights() const { return num_lights; }

    __host__ __device__ inline Light *getLight(int i) const {
        assert(i >= 0 && i < num_lights);
        return lights[i];
    }

    __host__ __device__ inline Group *getGroup() const { return group; }

    friend class SceneParser;

private:
    Camera *camera = nullptr;
    Vector3f background_color = Vector3f(0.5, 0.5, 0.5);
    Vector3f environment_color = Vector3f::ZERO;
    int num_lights = 0;
    Light **lights = nullptr;
    Group *group = nullptr;
};
