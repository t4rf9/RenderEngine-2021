#pragma once

#include <curand_kernel.h>

#include <vecmath.h>

#include "camera/cameras.h"
#include "hit.h"
#include "image.h"
#include "lights/lights.h"
#include "material.h"
#include "objects/group.h"
#include "ray.h"
#include "scene/scene.h"

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

__global__ void render(Image *image, Scene *scene, curandState *rand_state);
