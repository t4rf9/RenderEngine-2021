#pragma once

#include <vecmath.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_error.h"

// Simple image class
class Image {
public:
    Image(int w, int h) {
        width = w;
        height = h;
        data = new Vector3f[width * height];
    }

    ~Image() { delete[] data; }

    static void *operator new(std::size_t sz);

    static void *operator new[](std::size_t sz);

    static void operator delete(void *ptr);

    static void operator delete[](void *ptr);

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
