#include "revsurface.h"

std::default_random_engine RevSurface::generator;

std::uniform_real_distribution<double> RevSurface::real_0_1_distribution{0, 1};

std::uniform_real_distribution<double> RevSurface::real_10_distribution{-10, 10};

RevSurface::RevSurface(Curve *pCurve, Material *material) : Object3D(material), pCurve(pCurve) {
    auto *curve_box = pCurve->get_bounding_box();
    if (curve_box != nullptr) {
        const auto &curve_min = curve_box->get_min();
        const auto &curve_max = curve_box->get_max();

        // Check flat.
        if (curve_min.z() != 0 || curve_max.z() != 0) {
            printf("Profile of revSurface must be flat on xy plane.\n");
            exit(0);
        }

        Vector3f max = curve_max;
        if (max.x() < -curve_min.x()) {
            max.x() = -curve_min.x();
        }
        max.z() = max.x();

        Vector3f min = Vector3f(-max.x(), curve_min.y(), -max.z());

        std::cout << "RevSurface bouning box:\t(" << min[0] << ", " << min[1] << ", " << min[2]
                  << ")\t(" << max[0] << ", " << max[1] << ", " << max[2] << ")" << std::endl;

        pBound = new BoundingBox(min, max);
    } else {
        // Check flat.
        for (const auto &cp : pCurve->getControls()) {
            if (cp.z() != 0.0) {
                printf("Profile of revSurface must be flat on xy plane.\n");
                exit(0);
            }
        }
    }
}

RevSurface::~RevSurface() {
    delete pCurve;
    delete pBound;
}

const int repeat_limit = 1000;
const int iterate_limit = 500;

bool RevSurface::intersect(const Ray &ray, Hit &hit, float t_min) {
    // (PA3 optional TODO): implement this for the ray-tracing routine using G-N
    // iteration.
    Vector3f d = ray.getDirection();
    Vector3f o = ray.getOrigin();

    // first check intersection with bounding object
    if (!pBound->intersect(ray, t_min)) {
        // std::cout << "\tbounding box not intersecting" << std::endl;
        // std::cout << "\t\t" << ray << std::endl;
        return false;
    }

    bool res = false;

    for (int i = 0; i < repeat_limit; i++) {
        Vector3f x(t_min + real_0_1_distribution(generator), real_0_1_distribution(generator),
                   real_10_distribution(generator));
        auto &t = x[0];
        auto &u = x[1];
        auto &v = x[2];

        int count = 0;
        while (count++ < iterate_limit && 0 <= u && u <= 1 && t >= t_min) {
            float v2 = v * v;
            float divisor = 1 + v2;
            float sinv = 2 * v / divisor;
            float cosv = (1 - v2) / divisor;
            CurvePoint f = pCurve->curve_point_at_t(u);
            Vector3f F = o + t * d - Vector3f(cosv * f.V.x(), f.V.y(), sinv * f.V.x());

            Vector3f T_y = Vector3f(cosv * f.T.x(), f.T.y(), sinv * f.T.x());
            if (F.length() < 1e-6) {
                if (t <= t_min || t >= hit.getT()) {
                    break;
                }
                Vector3f T_xz = Vector3f(-sinv, 0, cosv);
                hit.set(t, material, Vector3f::cross(T_xz, T_y.normalized()));
                res = true;
                break;
            }

            Matrix3f F_prime = Matrix3f(
                d, -T_y,
                Vector3f((2 * sinv / divisor) * f.V[0], 0, -((2 * cosv) / divisor) * f.V[0]));
            Vector3f dx = F_prime.inverse() * F;

            x -= dx;
        }
    }
    return res;
}

/*
const int repeat_limit = 1000;
const int iterate_limit = 500;

bool RevSurface::intersect(const Ray &ray, Hit &hit, float t_min) {
    // (PA3 optional TODO): implement this for the ray-tracing routine using G-N
    // iteration.
    // std::cout << "RevSurface::intersect" << std::endl;
    // std::cout << "\t" << ray << std::endl;
    Vector3f d = ray.getDirection();
    Vector3f o = ray.getOrigin();
    bool res = false;
    for (int i = 0; i < repeat_limit; i++) {
        float u = real_0_1_distribution(generator);
        int count = 0;
        if (d.y() != 0) {
            while (count++ < iterate_limit && 0 <= u && u <= 1) {
                CurvePoint f = pCurve->curve_point_at_t(u);

                float _a = f.V.y() - o.y();
                float _b1 = o.x() * d.y() + _a * d.x();
                float _b2 = o.z() * d.y() + _a * d.z();
                float _b3 = d.y() * f.V.x();
                float F = _b1 * _b1 + _b2 * _b2 - _b3 * _b3;
                if (fabs(F) < 1e-5) {
                    float t = _a / d.y();

                    if (t <= t_min || t >= hit.getT()) {
                        break;
                    }

                    float cosv = _b1 / _b3;
                    float sinv = _b2 / _b3;

                    if (fabs(cosv * cosv + sinv * sinv - 1) > 8e-2) {
                        break;
                    }

                    Vector3f T_xz = Vector3f(-sinv, 0, cosv);
                    Vector3f T_y = Vector3f(f.T.x() * cosv, f.T.y(), f.T.x() * sinv);
                    hit.set(t, material, Vector3f::cross(T_y.normalized(), T_xz));

                    // std::cout << "\tu:\t" << u << ",\tt:\t" << t << std::endl;
                    // std::cout << "\t\tcosv:\t" << cosv << ",\tsinv:\t" << sinv
                    //          << ",\tfabs(cosv * cosv + sinv * sinv - 1):\t"
                    //          << fabs(cosv * cosv + sinv * sinv - 1) << std::endl;
                    // std::cout << "\t\t" << hit << std::endl;
                    // Vector3f rp = ray.pointAtParameter(t);
                    // Vector3f cp = Vector3f(f.V.x() * cosv, f.V.y(), f.V.x() * sinv);
                    // std::cout << "\t\tray:\t" << rp[0] << ", " << rp[1] << ", " << rp[2]
                    //          << std::endl;
                    // std::cout << "\t\tcurve:\t" << cp[0] << ", " << cp[1] << ", " << cp[2]
                    //          << std::endl;

                    res = true;
                    break;
                }

                float F_prime = 2 * (f.T.y() * (_b1 * d.x() + _b2 * d.z()) - _b3 * d.y() * f.T.x());

                u -= F / F_prime;
            }
        } else {
            while (count++ < iterate_limit && 0 <= u && u <= 1) {
                CurvePoint f = pCurve->curve_point_at_t(u);
                float F = f.V.y() - o.y();
                if (fabs(F) < 1e-5) {
                    Vector2f o_2 = o.xz();
                    Vector2f Rd_2 = d.xz().normalized();
                    Vector2f l_2 = -o_2;
                    float l_2_length2 = l_2.absSquared();
                    float r = f.V.x();
                    float r2 = r * r;
                    bool inside = l_2_length2 < r2;
                    bool on = l_2_length2 == r2;

                    float tp = Vector2f::dot(l_2, Rd_2);
                    if (!inside && tp <= 0) {
                        break;
                    }

                    float d2 = l_2_length2 - tp * tp;
                    if (d2 > r2) {
                        break;
                    }

                    float dt = std::sqrt(r2 - d2);

                    float t = tp + ((inside || on) ? dt : -dt);
                    if (t <= t_min || t > hit.getT()) {
                        break;
                    }

                    float cosv = (o.x() + t * d.x()) / f.V.x();
                    float sinv = (o.z() + t * d.z()) / f.V.z();

                    if (fabs(cosv * cosv + sinv * sinv - 1) > 8e-2) {
                        break;
                    }

                    Vector3f T_xz = Vector3f(-sinv, 0, cosv);
                    Vector3f T_y = Vector3f(f.T.x() * cosv, f.T.y(), f.T.x() * sinv);
                    hit.set(t, material, Vector3f::cross(T_y.normalized(), T_xz));

                    // std::cout << "\t" << hit << std::endl;
                    // Vector3f rp = ray.pointAtParameter(t);
                    // Vector3f cp = Vector3f(f.V.x() * cosv, f.V.y(), f.V.x() * sinv);
                    // std::cout << "\t\tray:\t" << rp[0] << ", " << rp[1] << ", " << rp[2]
                    //          << std::endl;
                    // std::cout << "\t\tcurve:\t" << cp[0] << ", " << cp[1] << ", " << cp[2]
                    //          << std::endl;

                    res = true;
                    break;
                }

                float F_prime = f.T.y();

                u -= F / F_prime;
            }
        }
    }

    return res;
}
*/
