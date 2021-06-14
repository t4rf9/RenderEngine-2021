#pragma once

#include <curand_kernel.h>

#include <vecmath.h>

#include "device/camera/cameras.h"
#include "device/hit.h"
#include "device/lights/lights.h"
#include "device/material.h"
#include "device/objects/group.h"
#include "device/ray.h"
#include "image.h"
#include "scene/scene.h"

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

__global__ void render(Image *image, Scene **p_scene);
