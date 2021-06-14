#pragma once

#include <vecmath.h>

#include "scene/parameters/parameters.h"
#include "scene/scene.h"

__global__ void create_scene(Scene **p_scene, CameraParams *camera_params,
                             LightsParams *lights_params, GroupParams *base_group_params,
                             MaterialsParams *material_params, Vector3f background_color,
                             Vector3f environment_color);