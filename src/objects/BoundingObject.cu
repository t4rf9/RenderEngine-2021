#include "objects/BoundingObject.h"

void *BoundingObject::operator new(std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void *BoundingObject::operator new[](std::size_t sz) {
    void *res;
    checkCudaErrors(cudaMallocManaged(&res, sz));
    return res;
}

void BoundingObject::operator delete(void *ptr) { checkCudaErrors(cudaFree(ptr)); }

void BoundingObject::operator delete[](void *ptr) { checkCudaErrors(cudaFree(ptr)); }
