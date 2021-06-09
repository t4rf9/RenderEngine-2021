#include "scene.h"

#include "camera/cameras.h"
#include "lights/lights.h"
#include "objects/group.h"

Scene::~Scene() {
    delete camera;
    delete group;

    for (int i = 0; i < num_lights; i++) {
        delete lights[i];
    }
    delete[] lights;
}

void *Scene::operator new(std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void *Scene::operator new[](std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void Scene::operator delete(void *ptr) { checkCudaErrors(cudaFree(ptr)); }

void Scene::operator delete[](void *ptr) { checkCudaErrors(cudaFree(ptr)); }
