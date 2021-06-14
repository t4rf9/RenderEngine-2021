#pragma once

#include <vecmath.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_error.h"

const bool debug = true;

// Simple image class
class Image {
public:
    Image(int w, int h) {
        width = w;
        height = h;
        checkCudaErrors(cudaMallocManaged(&data, width * height * sizeof(Vector3f)));
        if (debug) {
            printf("image->data:\t[0x%lx, 0x%lx)\n", data,
                   data + width * height * sizeof(Vector3f));
        }
    }

    ~Image() { checkCudaErrors(cudaFree(data)); }

    static void *operator new(std::size_t sz) {
        void *res;
        checkCudaErrors(cudaMallocManaged(&res, sz));
        if (debug) {
            printf("image:\t0x%lx\n", res);
        }
        return res;
    }

    static void *operator new[](std::size_t sz) {
        void *res;
        checkCudaErrors(cudaMallocManaged(&res, sz));
        return res;
    }

    static void operator delete(void *ptr) { checkCudaErrors(cudaFree(ptr)); }

    static void operator delete[](void *ptr) { checkCudaErrors(cudaFree(ptr)); }

    inline int Width() const { return width; }

    inline int Height() const { return height; }

    inline const Vector3f &GetPixel(int x, int y) const {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return data[y * width + x];
    }

    __host__ __device__ void SetAllPixels(const Vector3f &color);

    __host__ __device__ inline void SetPixel(int x, int y, const Vector3f &color) {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        data[y * width + x] = color;
    }

    static Image *LoadPPM(const char *filename);

    void SavePPM(const char *filename) const;

    static Image *LoadTGA(const char *filename);

    void SaveTGA(const char *filename) const;

    int SaveBMP(const char *filename);

    void SaveImage(const char *filename);

private:
    int width;
    int height;
    Vector3f *data;
};
