#pragma once

#include <cassert>
#include <vecmath.h>

#include "device/camera/cameras.h"
#include "device/hit.h"
#include "device/lights/lights.h"
#include "device/material.h"
#include "device/objects/BezierCurve.h"
#include "device/objects/BoundingBox.h"
#include "device/objects/BoundingObject.h"
#include "device/objects/BsplineCurve.h"
#include "device/objects/curve.h"
#include "device/objects/group.h"
#include "device/objects/mesh.h"
#include "device/objects/object3d.h"
#include "device/objects/plane.h"
#include "device/objects/revsurface.h"
#include "device/objects/sphere.h"
#include "device/objects/transform.h"
#include "device/objects/triangle.h"
#include "device/ray.h"
#include "parameters/parameters.h"

class Scene {
public:
    __device__ Scene() = delete;

    __device__ Scene(CameraParams *camera_params, LightsParams *lights_params,
                     GroupParams *base_group_params, MaterialsParams *materials_params,
                     Vector3f background_color, Vector3f environment_color);

    __device__ ~Scene();

    __device__ inline Camera *getCamera() const { return camera; }

    __device__ inline Vector3f getBackgroundColor() const { return background_color; }

    __device__ inline Vector3f getEnvironmentColor() const { return environment_color; }

    __device__ inline int getNumLights() const { return num_lights; }

    __device__ inline Light *getLight(int i) const {
        assert(i >= 0 && i < num_lights);
        return lights[i];
    }

    __device__ inline Group *getGroup() const { return base_group; }

private:
    Camera *camera = nullptr;

    Vector3f background_color = Vector3f(0.5f, 0.5f, 0.5f);
    Vector3f environment_color = Vector3f::ZERO;

    int num_lights = 0;
    Light **lights = nullptr;

    Group *base_group = nullptr;
    GroupParams *base_group_params; // for destruction

    int num_materials = 0;
    Material **materials = nullptr;
};
