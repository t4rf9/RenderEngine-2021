#include "light.h"

__device__ Light::Light(Type type) : type(type) {}

__device__ Light::~Light() {}