#include "destroy_scene.h"

__global__ void destroy_scene(Scene **p_scene) { delete *p_scene; }
