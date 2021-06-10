#include "scene.h"

#include "camera/cameras.h"
#include "lights/lights.h"
#include "objects/group.h"

Scene::~Scene() {
    delete camera;
    delete group;

    for (int i = 0; i < num_lights; i++) {
        delete lights[i];
    }
    delete[] lights;
}
