#include "random.h"

__device__ Vector2f Random::unit_disk() {
    while (true) {
        float a = 2.f * (*this)() - 1.f;
        float b = 2.f * (*this)() - 1.f;
        if (a * a + b * b <= 1) {
            return Vector2f(a, b);
        }
    }
}
