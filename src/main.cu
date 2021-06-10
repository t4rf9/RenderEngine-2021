#include "camera/cameras.h"
#include "image.h"
#include "lights/lights.h"
#include "objects/group.h"
#include "scene.h"
#include "scene_parser.h"

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

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

__global__ void render(Image *image, Scene *scene, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    Camera *camera = scene->getCamera();
    int width = camera->getWidth();

    if (x >= width || y >= camera->getHeight()) {
        return;
    }

    int pixel_index = y * width + x;
    curandState *local_rand_state = &rand_state[pixel_index];
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(2021, pixel_index, 0, local_rand_state);

    Group *baseGroup = scene->getGroup();

    // 计算当前像素(x,y)处相机出射光线ray
    Ray *ray = camera->generateRay(Vector2f(x, y));
    Hit hit;
    // 判断ray是否和场景有交点，并返回最近交点的数据，存储在hit中
    bool hasIntersection = baseGroup->intersect(*ray, hit, 0.f, local_rand_state);

    if (hasIntersection) {
        Vector3f finalColor = scene->getEnvironmentColor() * hit.getMaterial()->getDiffuseColor();

        const int refract_depth_limit = 10;
        const int reflect_depth_limit = 10;
        const int depth_limit =
            refract_depth_limit > reflect_depth_limit ? refract_depth_limit : reflect_depth_limit;
        Ray **Q = new Ray *[depth_limit + 1];
        int ql = 0;
        int qr = 0;
        while (true) {
            auto *material = hit.getMaterial();
            auto t = hit.getT();
            auto intersection = ray->pointAtParameter(t);
            auto depth = ray->get_depth();
            auto weight = ray->get_weight();
            auto incident_direction = ray->getDirection();
            auto incident_refractive_index = ray->get_incident_refractive_index();
            auto exit_refractive_index = material->get_refractive_index();
            Vector3f &normal = hit.getNormal_var();
            float cos_in = -Vector3f::dot(incident_direction, normal);
            if (cos_in < 0.f) {
                normal = -normal;
                cos_in = -cos_in;
            }
            Vector3f color = Vector3f(0.f);

            // 找到交点之后，累加来自所有光源的光强影响

            int num_lights = scene->getNumLights();
            for (int li = 0; li < num_lights; li++) {
                Light *light = scene->getLight(li);
                Vector3f L, lightColor;
                // 获得光照强度
                light->getIllumination(intersection, L, lightColor);
                float len = L.normalize();
                // shadow
                Hit tmp;
                hasIntersection =
                    shadow && baseGroup->intersect(Ray(intersection + 1e-3f * L, L, 0, 1.f, 1.f),
                                                   tmp, 0.f, local_rand_state);
                if (!shadow || !hasIntersection || tmp.getT() >= len) {
                    // 计算局部光强
                    color += material->Shade(*ray, hit, L, lightColor);
                }
            }
            color *= weight;
            finalColor += color;

            delete ray;

            // reflect
            if (reflect && material->reflective() && depth < reflect_depth_limit) {
                Vector3f reflect_direction = 2.f * cos_in * normal + incident_direction;
                Q[qr++] = new Ray(intersection + 1e-3f * reflect_direction, reflect_direction,
                                  depth + 1, weight * material->get_reflect_coefficient(),
                                  incident_refractive_index);
            }

            // refract
            if (refract && material->refractive() && depth < refract_depth_limit) {
                float refractive_index = incident_refractive_index / exit_refractive_index;
                float cos_out =
                    sqrt(1 - refractive_index * refractive_index * (1.f - cos_in * cos_in));
                Vector3f refract_direction = refractive_index * incident_direction +
                                             (refractive_index * cos_in - cos_out) * normal;

                Q[qr++] =
                    new Ray(intersection + 1e-3f * refract_direction, refract_direction, depth + 1,
                            weight * material->get_refract_coefficient(), exit_refractive_index);
            }

            if (ql == qr) {
                break;
            }
            while (ql != qr) {
                hit.clear();
                hasIntersection = baseGroup->intersect(*Q[ql], hit, 0.f, local_rand_state);
                if (hasIntersection) {
                    ray = Q[ql++];
                    break;
                }
                delete Q[ql++];
            }
            if (!hasIntersection) {
                break;
            }
        }

        for (int i = ql; i < qr; i++) {
            delete Q[i];
        }
        delete[] Q;

        image->SetPixel(x, y, finalColor);
    } else {
        // 不存在交点，返回背景色
        image->SetPixel(x, y, scene->getBackgroundColor());
    }
}

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
