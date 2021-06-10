#include "camera/cameras.h"
#include "image.h"
#include "lights/lights.h"
#include "objects/group.h"
#include "scene/scene.h"
#include "scene/scene_parser.h"
#include "render.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>

#include <curand_kernel.h>

#include "cuda_error.h"

int main(int argc, char *argv[]) {
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc != 3) {
        std::cout << "Usage: ./bin/main <input scene file> <output bmp file>" << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2]; // only bmp is allowed.

    // First, parse the scene using SceneParser.
    SceneParser *sceneParser = new SceneParser(inputFile.c_str());
    Scene *scene = sceneParser->getScene();
    Camera *camera = scene->getCamera();

    int width = camera->getWidth();
    int height = camera->getHeight();

    Image *image = new Image(width, height);

    // Then loop over each pixel in the image, shooting a ray
    // through that pixel and finding its intersection with
    // the scene.  Write the color at the intersection to that
    // pixel in your output image.

    // 循环屏幕空间的像素

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, width * height * sizeof(curandState)));

    dim3 block_size(8, 8);
    dim3 num_blocks((width + block_size.x - 1) / block_size.x,
                    (height + block_size.y - 1) / block_size.y);

    render<<<num_blocks, block_size>>>(image, scene, d_rand_state);
    /*for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            render(image, scene, x, y);
        }
    }*/
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_rand_state));

    image->SaveBMP(outputFile.c_str());

    delete image;
    delete sceneParser;

    return 0;
}
