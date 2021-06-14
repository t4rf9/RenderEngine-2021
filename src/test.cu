#include "device/camera/cameras.h"
#include "device/lights/lights.h"
#include "device/objects/BezierCurve.h"
#include "device/objects/BoundingBox.h"
#include "device/objects/BoundingObject.h"
#include "device/objects/group.h"
#include "image.h"
#include "scene/scene_parser.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>

__global__ void init(Camera **p_camera) {
    *p_camera = new PerspectiveCamera(Vector3f(0, 0, 10), Vector3f(0, 0, -1),
                                      Vector3f(0, 1, 0), 800, 600, float(M_PI) / 6.f);
}

__global__ void destroy(Camera **p_camera) { delete *p_camera; }

__global__ void test_kernel(Camera **p_camera) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    Camera *camera = *p_camera;
    Ray ray = camera->generateRay(Vector2f(x, y));
    Vector3f ooo = ray.getOrigin();
}

int main(int argc, char *argv[]) {
    Camera **p_camera;
    checkCudaErrors(cudaMallocManaged(&p_camera, sizeof(Camera **)));

    init<<<1, 1>>>(p_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    test_kernel<<<dim3(1000, 750), dim3(8, 8)>>>(p_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    destroy<<<1, 1>>>(p_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(p_camera));
    return 0;
}