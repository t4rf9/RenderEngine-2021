#include "perspective_camera.h"

__device__
PerspectiveCamera::PerspectiveCamera(const Vector3f &pos, const Vector3f &direction,
                                     const Vector3f &up, int imgW, int imgH, float angle,
                                     float focus_dst, float aperture)
    : Camera(pos, direction, up, imgW, imgH), angle(angle),
      center(float(imgW / 2), float(imgH / 2)), focus_dst(focus_dst), aperture(aperture),
      f(2.f * std::tan(angle * 0.5f) * focus_dst / float(imgH)),
      R(this->horizontal, this->up, this->direction) {}

__device__ Ray PerspectiveCamera::generateRay(const Vector2f &point, RandState &random) {
    Vector3f offset(random_unit_disk(random) * aperture, 0);
    float dx = point[0] - center[0];
    float dy = point[1] - center[1];
    Vector3f d_camera(dx * f, dy * f, focus_dst);
    d_camera -= offset;

    return Ray(pos + R * offset, (R * d_camera).normalized(), 0, 1.f, 1.f);
}

__device__ Vector2f PerspectiveCamera::random_unit_disk(RandState &random) {
    while (true) {
        float a = 2.f * curand_uniform(&random) - 1.f;
        float b = 2.f * curand_uniform(&random) - 1.f;
        if (a * a + b * b <= 1.f) {
            return Vector2f(a, b);
        }
    }
}
