#include "scene.h"

__device__ Scene::Scene(CameraParams *camera_params, LightsParams *lights_params,
                        GroupParams *base_group_params, MaterialsParams *materials_params,
                        Vector3f background_color, Vector3f environment_color)
    : background_color(background_color), environment_color(environment_color),
      base_group_params(base_group_params) {
    camera = new PerspectiveCamera(camera_params->pos, camera_params->direction,
                                   camera_params->up, camera_params->width,
                                   camera_params->height, camera_params->angle,
                                   camera_params->focus_dist, camera_params->aperture);

    num_lights = lights_params->num_lights;
    lights = new Light *[num_lights];
    for (int i = 0; i < num_lights; i++) {
        LightParams &light_params = lights_params->lights[i];
        switch (light_params.type) {
        case LightParams::Type::Point:
            lights[i] = new PointLight(light_params.position, light_params.color);
            break;
        case LightParams::Type::Directional:
            lights[i] = new DirectionalLight(light_params.direction, light_params.color);
            break;
        default:
            break;
        }
    }

    num_materials = materials_params->num_materials;
    materials = new Material *[num_materials];
    for (int i = 0; i < num_materials; i++) {
        MaterialParams &material_params = materials_params->materials[i];
        materials[i] = new Material(
            material_params.diffuseColor, material_params.specularColor,
            material_params.shininess, material_params.reflect_coefficient,
            material_params.refract_coefficient, material_params.refractive_index);
    }

    base_group = new Group(base_group_params->num_objects);

    struct GroupListNode {
        Group *group;
        GroupParams *group_params;
        GroupListNode *next;
    };

    GroupListNode *group_node = new GroupListNode;
    group_node->group = base_group;
    group_node->group_params = base_group_params;
    group_node->next = nullptr;

    GroupListNode **group_next = &group_node->next;

    while (group_node != nullptr) {
        Group *group = group_node->group;
        for (int i = 0; i < group_node->group_params->num_objects; i++) {
            ObjectParamsPointer &object_params = group_node->group_params->objects[i];
            Object3D *object;
            switch (group_node->group_params->object_types[i]) {
            case ObjectType::TRIANGLE:
                object = new Triangle(object_params.triangle->vertices[0],
                                      object_params.triangle->vertices[1],
                                      object_params.triangle->vertices[2],
                                      materials[object_params.triangle->material_id]);
                break;
            case ObjectType::SPHERE:
                object =
                    new Sphere(object_params.sphere->center, object_params.sphere->radius,
                               materials[object_params.sphere->material_id]);
                break;
            case ObjectType::PLANE:
                object = new Plane(object_params.plane->normal, object_params.plane->d,
                                   materials[object_params.plane->material_id]);
                break;
            case ObjectType::MESH:
                object = new Mesh(
                    object_params.mesh->vertices, object_params.mesh->num_vertices,
                    object_params.mesh->face_indices, object_params.mesh->num_faces,
                    materials[object_params.mesh->material_id]);
                break;
            case ObjectType::REVSURFACE:
                Curve *pCurve;
                switch (object_params.revsurface->curve->type) {
                case CurveParams::Type::Bezier:
                    pCurve =
                        new BezierCurve(object_params.revsurface->curve->controls,
                                        object_params.revsurface->curve->num_controls);
                    break;
                case CurveParams::Type::BSpline:
                    pCurve =
                        new BsplineCurve(object_params.revsurface->curve->controls,
                                         object_params.revsurface->curve->num_controls);
                    break;
                default:
                    pCurve = nullptr;
                    break;
                }
                object = new RevSurface(pCurve,
                                        materials[object_params.revsurface->material_id]);
                break;
            case ObjectType::TRANSFORM: {
                struct TransformListNode {
                    TransformParams *params;
                    TransformListNode *prev;
                    TransformListNode *next;
                };

                TransformListNode *transform_node = new TransformListNode;
                transform_node->prev = nullptr;
                transform_node->next = nullptr;
                transform_node->params = object_params.transform;

                // Traverse nested transformations.
                // We require there is a object under the lowest level of Transform.
                while (transform_node->params->object_type == ObjectType::TRANSFORM) {
                    transform_node->next = new TransformListNode;
                    TransformListNode *next = transform_node->next;
                    next->prev = transform_node;
                    next->next = nullptr;
                    next->params = transform_node->params->object.transform;

                    transform_node = next;
                }

                ObjectParamsPointer &transformed_object_params =
                    transform_node->params->object;
                Object3D *transformed_object; // under lowest level of transform

                // Construct the final transformed object.
                switch (transform_node->params->object_type) {
                case ObjectType::TRIANGLE:
                    transformed_object = new Triangle(
                        transformed_object_params.triangle->vertices[0],
                        transformed_object_params.triangle->vertices[1],
                        transformed_object_params.triangle->vertices[2],
                        materials[transformed_object_params.triangle->material_id]);
                    break;
                case ObjectType::SPHERE:
                    transformed_object = new Sphere(
                        transformed_object_params.sphere->center,
                        transformed_object_params.sphere->radius,
                        materials[transformed_object_params.sphere->material_id]);
                    break;
                case ObjectType::PLANE:
                    transformed_object = new Plane(
                        transformed_object_params.plane->normal,
                        transformed_object_params.plane->d,
                        materials[transformed_object_params.plane->material_id]);
                    break;
                case ObjectType::MESH:
                    transformed_object =
                        new Mesh(transformed_object_params.mesh->vertices,
                                 transformed_object_params.mesh->num_vertices,
                                 transformed_object_params.mesh->face_indices,
                                 transformed_object_params.mesh->num_faces,
                                 materials[transformed_object_params.mesh->material_id]);
                    break;
                case ObjectType::REVSURFACE: {
                    Curve *pCurve;
                    switch (transformed_object_params.revsurface->curve->type) {
                    case CurveParams::Type::Bezier:
                        pCurve = new BezierCurve(
                            transformed_object_params.revsurface->curve->controls,
                            transformed_object_params.revsurface->curve->num_controls);
                        break;
                    case CurveParams::Type::BSpline:
                        pCurve = new BsplineCurve(
                            transformed_object_params.revsurface->curve->controls,
                            transformed_object_params.revsurface->curve->num_controls);
                        break;
                    default:
                        pCurve = nullptr;
                        break;
                    }
                    transformed_object = new RevSurface(
                        pCurve,
                        materials[transformed_object_params.revsurface->material_id]);
                    break;
                }
                case ObjectType::GROUP:
                    *group_next = new GroupListNode;
                    (*group_next)->next = nullptr;
                    (*group_next)->group_params = transformed_object_params.group;
                    (*group_next)->group =
                        new Group(transformed_object_params.group->num_objects);
                    transformed_object = (*group_next)->group;
                    group_next = &(*group_next)->next;
                    break;
                default:
                    transformed_object = nullptr;
                    break;
                }

                while (transform_node != nullptr) {
                    transformed_object =
                        new Transform(transform_node->params->matrix, transformed_object);
                    TransformListNode *tmp = transform_node;
                    transform_node = transform_node->prev;
                    delete tmp;
                }

                object = transformed_object;
                break;
            }
            case ObjectType::GROUP:
                *group_next = new GroupListNode;
                (*group_next)->next = nullptr;
                (*group_next)->group_params = object_params.group;
                (*group_next)->group = new Group(object_params.group->num_objects);
                object = (*group_next)->group;
                group_next = &(*group_next)->next;
                break;
            default:
                object = nullptr;
                break;
            }
            group->addObject(i, object);
        }

        GroupListNode *tmp = group_node;
        group_node = group_node->next;
        delete tmp;
    }
}

__device__ Scene::~Scene() {
    delete camera;

    for (int i = 0; i < num_lights; i++) {
        delete lights[i];
    }
    delete[] lights;

    for (int i = 0; i < num_materials; i++) {
        delete materials[i];
    }
    delete[] materials;

    struct GroupListNode {
        Group *group;
        GroupParams *group_params;
        GroupListNode *next;
    };

    GroupListNode *group_node = new GroupListNode;
    group_node->group = base_group;
    group_node->group_params = base_group_params;
    group_node->next = nullptr;

    GroupListNode **group_next = &group_node->next;

    while (group_node != nullptr) {
        Group *group = group_node->group;
        for (int i = 0; i < group->getGroupSize(); i++) {
            ObjectParamsPointer &object_params = group_node->group_params->objects[i];
            Object3D *object = group->getObject(i);

            switch (group_node->group_params->object_types[i]) {
            case ObjectType::TRANSFORM: {
                Transform *transform = static_cast<Transform *>(object);
                TransformParams *transform_params = object_params.transform;
                while (transform_params->object_type == ObjectType::TRANSFORM) {
                    Transform *tmp = transform;
                    transform = static_cast<Transform *>(transform->getObject());
                    delete tmp;
                }

                object = transform->getObject();
                switch (transform_params->object_type) {
                case ObjectType::GROUP:
                    *group_next = new GroupListNode;
                    (*group_next)->next = nullptr;
                    (*group_next)->group_params = transform_params->object.group;
                    (*group_next)->group = static_cast<Group *>(object);
                    group_next = &(*group_next)->next;
                    break;
                default:
                    delete object;
                    break;
                }

                delete transform;
                break;
            }
            case ObjectType::GROUP:
                *group_next = new GroupListNode;
                (*group_next)->next = nullptr;
                (*group_next)->group_params = object_params.group;
                (*group_next)->group = static_cast<Group *>(object);
                group_next = &(*group_next)->next;
                break;
            default:
                delete object;
                break;
            }
        }

        delete group;

        GroupListNode *tmp = group_node;
        group_node = group_node->next;
        delete tmp;
    }
}
