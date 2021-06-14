#include "material.h"

Material::Material(const Vector3f &d_color, const Vector3f &s_color, float shininess,
                   float reflect_coefficient, float refract_coefficient, float refractive_index)
    : diffuseColor(d_color), specularColor(s_color), shininess(shininess),
      reflect_coefficient(reflect_coefficient), refract_coefficient(refract_coefficient),
      refractive_index(refractive_index) {}

Vector3f Material::getSpecularColor() const { return specularColor; }

Vector3f Material::getDiffuseColor() const { return diffuseColor; }

Vector3f Material::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight,
                         const Vector3f &lightColor) {
    Vector3f shaded = lightColor;
    Vector3f N = hit.getNormal();

    Vector3f diffuse = diffuseColor * clamp(Vector3f::dot(dirToLight, N));

    Vector3f V = -ray.getDirection();
    Vector3f R = 2 * Vector3f::dot(N, dirToLight) * N - dirToLight;
    Vector3f specular = specularColor * pow(clamp(Vector3f::dot(V, R)), shininess);

    shaded *= diffuse + specular;

    return shaded;
}
