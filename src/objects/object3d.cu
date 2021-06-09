#include "object3d.h"

Object3D::Object3D() : material(nullptr) {}

Object3D::Object3D(Material *material) : material(material) {}

void *Object3D::operator new(std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void *Object3D::operator new[](std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void Object3D::operator delete(void *ptr) { checkCudaErrors(cudaFree(ptr)); }

void Object3D::operator delete[](void *ptr) { checkCudaErrors(cudaFree(ptr)); }
