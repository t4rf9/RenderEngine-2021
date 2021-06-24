#include "define.h"

#include "create_scene.h"
#include "destroy_scene.h"
#include "device/camera/cameras.h"
#include "device/lights/lights.h"
#include "device/spaces/group.h"
#include "image.h"
#include "render.h"
#include "scene/scene.h"
#include "scene/scene_parser.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>

#include <curand_kernel.h>
#include <vecmath.h>

#include "cuda_error.h"

int main(int argc, char *argv[]) {
    
    Quat4f rot;
    rot.setAxisAngle(M_PI / 2.f, Vector3f(0, -1, 0));

    (Matrix3f::rotation(rot) * Vector3f(1, 1, 0)).print();

    return 0;
}
