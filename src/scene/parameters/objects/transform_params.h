#pragma once

#include <vecmath.h>

#include "object_params_pointer.h"
#include "object_type.h"

struct GroupParams;

struct TransformParams {
    Matrix4f matrix;
    ObjectParamsPointer object;
    ObjectType object_type;
};
