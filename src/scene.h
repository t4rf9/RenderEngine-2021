#pragma once

#include <cassert>
#include <vecmath.h>

class Camera;
class Light;
class Group;

class Scene {
public:
    Scene() = default;

    ~Scene();

    inline Camera *getCamera() const { return camera; }

    inline Vector3f getBackgroundColor() const { return background_color; }

    inline Vector3f getEnvironmentColor() const { return environment_color; }

    inline int getNumLights() const { return num_lights; }

    inline Light *getLight(int i) const {
        assert(i >= 0 && i < num_lights);
        return lights[i];
    }

    inline Group *getGroup() const { return group; }

    friend class SceneParser;

private:
    Camera *camera = nullptr;
    Vector3f background_color = Vector3f(0.5, 0.5, 0.5);
    Vector3f environment_color = Vector3f::ZERO;
    int num_lights = 0;
    Light **lights = nullptr;
    Group *group = nullptr;
};
