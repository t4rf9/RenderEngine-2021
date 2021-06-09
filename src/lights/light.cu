#include "lights/light.h"

void *Light::operator new(std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void *Light::operator new[](std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void Light::operator delete(void *ptr) { checkCudaErrors(cudaFree(ptr)); }

void Light::operator delete[](void *ptr) { checkCudaErrors(cudaFree(ptr)); }
