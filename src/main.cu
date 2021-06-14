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

#include "cuda_error.h"

const bool shadow = true;
const bool reflect = true;
const bool refract = true;

//__global__
void render(Image *image, Scene *scene, int x, int y) {
    // std::cout << "(" << x << ", " << y << ")" << std::endl;

    Camera *camera = scene->getCamera();
    Group *baseGroup = scene->getGroup();

    // 计算当前像素(x,y)处相机出射光线ray

    Ray ray = camera->generateRay(Vector2f(x, y));
    Hit hit;
    uint_fast32_t rand = 1;
    // 判断ray是否和场景有交点，并返回最近交点的数据，存储在hit中
    bool hasIntersection = baseGroup->intersect(ray, hit, 0.f, rand);

    if (hasIntersection) {
        Vector3f finalColor = scene->getEnvironmentColor() * hit.getMaterial()->getDiffuseColor();
        // Vector3f finalColor = Vector3f::ZERO;
        std::queue<Ray> Q;
        while (true) {
            auto *material = hit.getMaterial();
            auto t = hit.getT();
            auto intersection = ray.pointAtParameter(t);
            auto depth = ray.get_depth();
            auto weight = ray.get_weight();
            auto incident_direction = ray.getDirection();
            auto incident_refractive_index = ray.get_incident_refractive_index();
            auto exit_refractive_index = material->get_refractive_index();
            Vector3f &normal = hit.getNormal_var();
            float cos_in = -Vector3f::dot(incident_direction, normal);
            if (cos_in < 0) {
                normal = -normal;
                cos_in = -cos_in;
            }
            Vector3f color = Vector3f::ZERO;

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
                    shadow &&
                    baseGroup->intersect(Ray(intersection + 1e-3 * L, L, 0, 1, 1), tmp, 0, rand);
                if (!shadow || !hasIntersection || tmp.getT() >= len) {
                    // 计算局部光强
                    color += material->Shade(ray, hit, L, lightColor);
                }
            }
            color *= weight;
            finalColor += color;

            // reflect
            if (reflect && material->reflective() && depth < 2) {
                Vector3f reflect_direction = 2 * cos_in * normal + incident_direction;
                Q.push(Ray(intersection + 1e-3 * reflect_direction, reflect_direction, depth + 1,
                           weight * material->get_reflect_coefficient(),
                           incident_refractive_index));
            }

            // refract
            if (refract && material->refractive() && depth < 2) {
                float refractive_index = incident_refractive_index / exit_refractive_index;
                float cos_out =
                    sqrt(1 - refractive_index * refractive_index * (1 - cos_in * cos_in));
                Vector3f refract_direction = refractive_index * incident_direction +
                                             (refractive_index * cos_in - cos_out) * normal;

                Q.push(Ray(intersection + 1e-3 * refract_direction, refract_direction, depth + 1,
                           weight * material->get_refract_coefficient(), exit_refractive_index));
            }

            if (Q.empty()) {
                break;
            }
            while (!Q.empty()) {
                hit.clear();
                hasIntersection = baseGroup->intersect(Q.front(), hit, 0, rand);
                if (hasIntersection) {
                    ray = Q.front();
                    Q.pop();
                    break;
                }
                Q.pop();
            }
            if (!hasIntersection) {
                break;
            }
        }
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

    Image *image = new Image(camera->getWidth(), camera->getHeight());

    // Then loop over each pixel in the image, shooting a ray
    // through that pixel and finding its intersection with
    // the scene.  Write the color at the intersection to that
    // pixel in your output image.

    // 循环屏幕空间的像素

    int width = camera->getWidth();
    int height = camera->getHeight();

    dim3 block_size(8, 8);
    dim3 num_blocks((width + block_size.x - 1) / block_size.x,
                    (height + block_size.y - 1) / block_size.y);

    for (int x = 0; x < camera->getWidth(); x++) {
        for (int y = 0; y < camera->getHeight(); y++) {
            // render<<<num_blocks, block_size>>>(image, scene);
            render(image, scene, x, y);
        }
    }
    image->SaveBMP(outputFile.c_str());

    delete image;
    delete sceneParser;

    return 0;
}
