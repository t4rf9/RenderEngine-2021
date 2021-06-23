#include "define.h"

#include "create_scene.h"
#include "destroy_scene.h"
#include "device/camera/cameras.h"
#include "device/lights/lights.h"
#include "device/spaces/group.h"
#include "image.h"
#include "render.h"
#include "scene/scene.h"
#include "scene/scene_parser.h"

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
    // Shell arguments parsing.
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }
    if (argc != 3) {
        std::cout << "Usage: ./bin/main <input scene file> <output bmp file>"
                  << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2]; // only bmp is allowed.

    // First, parse the scene using SceneParser.
    SceneParser *sceneParser = new SceneParser(inputFile.c_str());
    if (debug)
        printf("sceneParser:\t0x%lx\n", sceneParser);

    // Allocate space for the scene.
    Scene **p_scene;
    checkCudaErrors(cudaMalloc(&p_scene, sizeof(Scene *)));
    if (debug)
        printf("p_scene:\t0x%lx\n", p_scene);

    // Construct the image buffer.
    CameraParams *camera_params = sceneParser->getCameraParams();
    int width = camera_params->width;
    int height = camera_params->height;
    Image *image = new Image(camera_params->width, camera_params->height);

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    // Create the scene on GPU.
    printf("create_scene:\t");
    clock_t start = clock();
    create_scene<<<1, 1>>>(
        p_scene, camera_params, sceneParser->getLightsParams(),
        sceneParser->getBaseGroupParams(), sceneParser->getMaterialsParams(),
        sceneParser->getBackgroundColor(), sceneParser->getEnvironmentColor());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("%lf s\n", double(clock() - start) / CLOCKS_PER_SEC);

    // Then loop over each pixel in the image, shooting a ray
    // through that pixel and finding its intersection with
    // the scene.  Write the color at the intersection to that
    // pixel in your output image.
    // 循环屏幕空间的像素
    dim3 block_size(8, 8);
    dim3 num_blocks((width + block_size.x - 1) / block_size.x,
                    (height + block_size.y - 1) / block_size.y);
    printf("render:\t\t");
    start = clock();
    render<<<num_blocks, block_size, 49152>>>(image, p_scene);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("%lf s\n", double(clock() - start) / CLOCKS_PER_SEC);

    // Delete the scene on GPU.
    printf("destroy_scene:\t");
    start = clock();
    destroy_scene<<<1, 1>>>(p_scene);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("%lf s\n", double(clock() - start) / CLOCKS_PER_SEC);

    // Free GPU only resources.

    printf("free p_scene\n");
    checkCudaErrors(cudaFree(p_scene));

    printf("free sceneParser\n");
    delete sceneParser;

    // Save the image.
    printf("save bmp\n");
    image->SaveBMP(outputFile.c_str());

    // Free the image buffer.
    delete image;

    // For leak checking.
    cudaDeviceReset();

    printf("done\n");

    return 0;
}
