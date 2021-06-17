#include "render.h"

__global__ void render(Image *image, Scene **p_scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    Scene *scene = *p_scene;

    Camera *camera = scene->getCamera();
    int width = camera->getWidth();

    if (x >= width || y >= camera->getHeight()) {
        return;
    }

    int pixel_index = y * width + x;

    RandState local_rand_state;
    // Each thread gets same seed, a different sequence number, no offset
    // curand_init(rand_seed, pixel_index, 0, &local_rand_state);

    Group *baseGroup = scene->getGroup();

    Vector3f finalColor;
    for (int i = 0; i < rays_per_pixel; i++) {
        // 计算当前像素(x,y)处相机出射光线ray
        Ray ray = camera->generateRay(
            Vector2f((x - 0.5) + local_rand_state(), (y - 0.5) + local_rand_state()));
        Hit hit;
        // 判断ray是否和场景有交点，并返回最近交点的数据，存储在hit中
        bool hasIntersection = baseGroup->intersect(ray, hit, 0.f, local_rand_state);

        if (hasIntersection) {
            finalColor +=
                scene->getEnvironmentColor() * hit.getMaterial()->getDiffuseColor();

            const int refract_depth_limit = 5;
            const int reflect_depth_limit = 5;
            const int depth_limit =
                (refract_depth_limit > reflect_depth_limit ? refract_depth_limit
                                                           : reflect_depth_limit) +
                1;
            Ray Q[depth_limit];
            int q_top = 0;
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
                    int shadow_check_time =
                        shadow && light->type == Light::DISK ? 10 : 1;
                    Vector3f color_tmp;
                    for (int j = 0; j < shadow_check_time; j++) {
                        light->getIllumination(intersection, L, lightColor);
                        float len = L.normalize();
                        // shadow
                        bool shadowed =
                            shadow && baseGroup->intersect(
                                          Ray(intersection + 1e-2f * L, L, 0, 1.f, 1.f),
                                          0.f, len, local_rand_state);
                        if (!shadowed) {
                            // 计算局部光强
                            color_tmp += material->Shade(ray, hit, L, lightColor);
                        }
                    }
                    color += color_tmp / shadow_check_time;
                }
                color *= weight;
                finalColor += color;

                // reflect
                if (reflect && material->reflective() && depth < reflect_depth_limit &&
                    weight > 1e-2f) {
                    Vector3f reflect_direction =
                        2.f * cos_in * normal + incident_direction;
                    Q[q_top++].set(intersection + 1e-3f * reflect_direction,
                                   reflect_direction, depth + 1,
                                   weight * material->get_reflect_coefficient(),
                                   incident_refractive_index);
                }

                // refract
                if (refract && material->refractive() && depth < refract_depth_limit &&
                    weight > 1e-2f) {
                    float refractive_index =
                        incident_refractive_index / exit_refractive_index;
                    float cos_out = sqrt(1.f - refractive_index * refractive_index *
                                                   (1.f - cos_in * cos_in));
                    Vector3f refract_direction =
                        refractive_index * incident_direction +
                        (refractive_index * cos_in - cos_out) * normal;

                    Q[q_top++].set(intersection + 1e-3f * refract_direction,
                                   refract_direction, depth + 1,
                                   weight * material->get_refract_coefficient(),
                                   exit_refractive_index);
                }

                hasIntersection = false;
                while (q_top > 0) {
                    hit.clear();
                    q_top--;
                    hasIntersection =
                        baseGroup->intersect(Q[q_top], hit, 0.f, local_rand_state);
                    if (hasIntersection) {
                        ray = Q[q_top];
                        break;
                    }
                }
                if (!hasIntersection) {
                    break;
                }
            }
        } else {
            // 不存在交点，返回背景色
            finalColor += scene->getBackgroundColor();
        }
    }

    image->SetPixel(x, y, finalColor / rays_per_pixel);
}
