#pragma once

#include "curve_params.h"
#include "mesh_params.h"
#include "object_params_pointer.h"
#include "object_type.h"
#include "plane_params.h"
#include "revsurface_params.h"
#include "sphere_params.h"
#include "transform_params.h"
#include "triangle_params.h"

struct GroupParams {
    ObjectParamsPointer *objects;
    ObjectType *object_types;
    int num_objects;
};
