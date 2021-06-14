#include "create_scene.h"

__global__ void create_scene(Scene **p_scene, CameraParams *camera_params,
                             LightsParams *lights_params, GroupParams *base_group_params,
                             MaterialsParams *material_params, Vector3f background_color,
                             Vector3f environment_color) {
    *p_scene = new Scene(camera_params, lights_params, base_group_params, material_params,
                         background_color, environment_color);
}