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

    __host__ __device__ inline Camera *getCamera() const { return camera; }

    __device__ inline Vector3f getBackgroundColor() const { return background_color; }

    __device__ inline Vector3f getEnvironmentColor() const { return environment_color; }

    __device__ inline int getNumLights() const { return num_lights; }

    __device__ inline Light *getLight(int i) const {
        assert(i >= 0 && i < num_lights);
        return lights[i];
    }

    __device__ inline Group *getGroup() const { return group; }

    friend class SceneParser;

private:
    Camera *camera = nullptr;

    Vector3f background_color = Vector3f(0.5f, 0.5f, 0.5f);
    Vector3f environment_color = Vector3f::ZERO;

    int num_lights = 0;
    Light **lights = nullptr;

    Group *group = nullptr;

    int num_materials;
    Material **materials;
};
