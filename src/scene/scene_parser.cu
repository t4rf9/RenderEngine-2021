#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "scene_parser.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DegreesToRadians(x) (M_PI * (x) / 180.0f)

SceneParser::SceneParser(const char *filename) {
    // initialize some reasonable default values

    checkCudaErrors(cudaMallocManaged(&camera_params, sizeof(CameraParams)));
    if (debug)
        printf("camera_params:\t0x%lx\n", camera_params);

    checkCudaErrors(cudaMallocManaged(&lights_params, sizeof(LightsParams)));
    lights_params->num_lights = 0;
    if (debug)
        printf("lights_params:\t0x%lx\n", lights_params);

    checkCudaErrors(cudaMallocManaged(&materials_params, sizeof(MaterialsParams)));
    materials_params->num_materials = 0;
    if (debug)
        printf("materials_params:\t0x%lx\n", materials_params);

    // parse the file
    assert(filename != nullptr);
    const char *ext = &filename[strlen(filename) - 4];

    if (strcmp(ext, ".txt") != 0) {
        printf("wrong file name extension\n");
        exit(0);
    }
    file = fopen(filename, "r");

    if (file == nullptr) {
        printf("cannot open scene file\n");
        exit(0);
    }
    parseFile();
    fclose(file);
    file = nullptr;

    if (lights_params->num_lights == 0) {
        printf("WARNING:    No lights specified\n");
    }
}

SceneParser::~SceneParser() {
    checkCudaErrors(cudaFree(camera_params));

    checkCudaErrors(cudaFree(lights_params->lights));
    checkCudaErrors(cudaFree(lights_params));

    for (int i = 0; i < materials_params->num_materials; i++) {
        if (materials_params->materials[i].texture != nullptr) {
            delete materials_params->materials[i].texture;
        }
    }
    checkCudaErrors(cudaFree(materials_params->materials));
    checkCudaErrors(cudaFree(materials_params));

    freeBaseGroupParams();
}

// ====================================================================
// ====================================================================

void SceneParser::parseFile() {
    if (debug)
        printf("SceneParser::parseFile()\n");
    //
    // at the top level, the scene can have a camera,
    // background color and a group of objects
    // (we add lights and other things in future assignments)
    //
    char token[MAX_PARSER_TOKEN_LENGTH];
    while (getToken(token)) {
        if (!strcmp(token, "PerspectiveCamera")) {
            parsePerspectiveCamera();
        } else if (!strcmp(token, "Background")) {
            parseBackground();
        } else if (!strcmp(token, "Lights")) {
            parseLights();
        } else if (!strcmp(token, "Materials")) {
            parseMaterials();
        } else if (!strcmp(token, "Group")) {
            checkCudaErrors(cudaMallocManaged(&base_group_params, sizeof(GroupParams)));
            if (debug)
                printf("base_group_params:\t0x%lx\n", base_group_params);
            parseGroup(base_group_params);
        } else {
            printf("Unknown token in parseFile: '%s'\n", token);
            exit(0);
        }
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parsePerspectiveCamera() {
    if (debug)
        printf("SceneParser::parsePerspectiveCamera()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the camera parameters
    getToken(token);
    assert(!strcmp(token, "{"));

    camera_params->pos = Vector3f::ZERO;
    camera_params->direction = Vector3f::FORWARD;
    camera_params->up = Vector3f::UP;
    camera_params->angle = DegreesToRadians(60.f);
    camera_params->width = 100;
    camera_params->height = 100;
    camera_params->focus_dist = 1.f;
    camera_params->aperture = 0.f;

    while (true) {
        getToken(token);
        if (!strcmp(token, "}")) {
            break;
        } else if (!strcmp(token, "center")) {
            camera_params->pos = readVector3f();
        } else if (!strcmp(token, "direction")) {
            camera_params->direction = readVector3f();
        } else if (!strcmp(token, "up")) {
            camera_params->up = readVector3f();
        } else if (!strcmp(token, "angle")) {
            float angle_degrees = readFloat();
            camera_params->angle = DegreesToRadians(angle_degrees);
        } else if (!strcmp(token, "width")) {
            camera_params->width = readInt();
        } else if (!strcmp(token, "height")) {
            camera_params->height = readInt();
        } else if (!strcmp(token, "focus_dist")) {
            camera_params->focus_dist = readFloat();
        } else if (!strcmp(token, "aperture")) {
            camera_params->aperture = readFloat();
        }
    }

    camera_params->type = CameraParams::Type::PerspectiveCamera;
}

void SceneParser::parseBackground() {
    if (debug)
        printf("SceneParser::parseBackground()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the background color
    getToken(token);
    assert(!strcmp(token, "{"));
    while (true) {
        getToken(token);
        if (!strcmp(token, "}")) {
            break;
        } else if (!strcmp(token, "color")) {
            background_color = readVector3f();
        } else if (!strcmp(token, "environment")) {
            environment_color = readVector3f();
        } else {
            printf("Unknown token in parseBackground: '%s'\n", token);
            assert(0);
        }
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parseLights() {
    if (debug)
        printf("SceneParser::parseLights()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    // read in the number of lights
    getToken(token);
    assert(!strcmp(token, "numLights"));
    lights_params->num_lights = readInt();
    checkCudaErrors(cudaMallocManaged(&lights_params->lights,
                                      lights_params->num_lights * sizeof(LightParams)));
    if (debug)
        printf("lights_params->lights:\t0x%lx\n", lights_params->lights);

    // read in the objects
    int count = 0;
    while (count < lights_params->num_lights) {
        getToken(token);
        if (strcmp(token, "DirectionalLight") == 0) {
            parseDirectionalLight(&lights_params->lights[count]);
        } else if (strcmp(token, "PointLight") == 0) {
            parsePointLight(&lights_params->lights[count]);
        } else if (strcmp(token, "DiskLight") == 0) {
            parseDiskLight(&lights_params->lights[count]);
        } else {
            printf("Unknown token in parseLight: '%s'\n", token);
            exit(0);
        }
        count++;
    }
    getToken(token);
    assert(!strcmp(token, "}"));
}

void SceneParser::parseDirectionalLight(LightParams *light_param) {
    if (debug)
        printf("SceneParser::parseDirectionalLight()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];

    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "direction"));
    light_param->direction = readVector3f();

    getToken(token);
    assert(!strcmp(token, "color"));
    light_param->color = readVector3f();

    getToken(token);
    assert(!strcmp(token, "}"));

    light_param->type = LightParams::Type::DIRECRIONAL;
}

void SceneParser::parsePointLight(LightParams *light_param) {
    if (debug)
        printf("SceneParser::parsePointLight()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];

    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "position"));
    light_param->position = readVector3f();

    getToken(token);
    assert(!strcmp(token, "color"));
    light_param->color = readVector3f();

    getToken(token);
    assert(!strcmp(token, "}"));

    light_param->type = LightParams::Type::POINT;
}

void SceneParser::parseDiskLight(LightParams *light_param) {
    if (debug)
        printf("SceneParser::parseDiskLight()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];

    light_param->position = Vector3f::ZERO;
    light_param->color = Vector3f::ZERO;
    light_param->radius = 0;
    light_param->normal = Vector3f(0, -1, 0);

    getToken(token);
    assert(!strcmp(token, "{"));

    while (true) {
        getToken(token);
        if (!strcmp(token, "}")) {
            break;
        } else if (!strcmp(token, "position")) {
            light_param->position = readVector3f();
        } else if (!strcmp(token, "color")) {
            light_param->color = readVector3f();
        } else if (!strcmp(token, "radius")) {
            light_param->radius = readFloat();
        } else if (!strcmp(token, "normal")) {
            light_param->normal = readVector3f().normalized();
        }
    }

    light_param->type = LightParams::Type::DISK;
}
// ====================================================================
// ====================================================================

void SceneParser::parseMaterials() {
    if (debug)
        printf("SceneParser::parseMaterials()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    // read in the number of materials
    getToken(token);
    assert(!strcmp(token, "numMaterials"));
    materials_params->num_materials = readInt();
    checkCudaErrors(
        cudaMallocManaged(&materials_params->materials,
                          materials_params->num_materials * sizeof(MaterialParams)));
    if (debug)
        printf("materials_params->materials:\t0x%lx\n", materials_params->materials);

    // read in the objects
    int count = 0;
    while (count < materials_params->num_materials) {
        getToken(token);
        if (!strcmp(token, "PhongMaterial")) {
            parsePhongMaterial(&materials_params->materials[count]);
        } else {
            printf("Unknown token in parsePhongMaterial: '%s'\n", token);
            exit(0);
        }
        count++;
    }

    getToken(token);
    assert(!strcmp(token, "}"));
}

void SceneParser::parsePhongMaterial(MaterialParams *material_params) {
    if (debug)
        printf("SceneParser::parsePhongMaterial()\n");
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];
    filename[0] = 0;

    material_params->diffuseColor = Vector3f(1.f, 1.f, 1.f);
    material_params->specularColor = Vector3f(0.f, 0.f, 0.f);
    material_params->shininess = 0.f;
    material_params->reflect_coefficient = 0.f;
    material_params->refract_coefficient = 0.f;
    material_params->refractive_index = 1.f;
    material_params->texture = nullptr;

    getToken(token);
    assert(!strcmp(token, "{"));

    while (true) {
        getToken(token);
        if (strcmp(token, "diffuseColor") == 0) {
            material_params->diffuseColor = readVector3f();
        } else if (strcmp(token, "specularColor") == 0) {
            material_params->specularColor = readVector3f();
        } else if (strcmp(token, "shininess") == 0) {
            material_params->shininess = readFloat();
        } else if (strcmp(token, "reflectCoefficient") == 0) {
            material_params->reflect_coefficient = readFloat();
        } else if (strcmp(token, "refractCoefficient") == 0) {
            material_params->refract_coefficient = readFloat();
        } else if (strcmp(token, "refractiveIndex") == 0) {
            material_params->refractive_index = readFloat();
        } else if (strcmp(token, "texture") == 0) {
            // read in texture and draw it.
            getToken(filename);

            const char *ext = &filename[strlen(filename) - 4];
            assert(!strcmp(ext, ".ppm"));

            material_params->texture = Image::LoadPPM(filename);
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parseObject(char token[MAX_PARSER_TOKEN_LENGTH],
                              ObjectParamsPointer *object, ObjectType *object_type) {
    if (!strcmp(token, "Group")) {
        *object_type = ObjectType::GROUP;
        checkCudaErrors(cudaMallocManaged(&object->group, sizeof(GroupParams)));
        parseGroup(object->group);
    } else if (!strcmp(token, "Sphere")) {
        *object_type = ObjectType::SPHERE;
        checkCudaErrors(cudaMallocManaged(&object->sphere, sizeof(SphereParams)));
        parseSphere(object->sphere);
    } else if (!strcmp(token, "Plane")) {
        *object_type = ObjectType::PLANE;
        checkCudaErrors(cudaMallocManaged(&object->plane, sizeof(PlaneParams)));
        parsePlane(object->plane);
    } else if (!strcmp(token, "Triangle")) {
        *object_type = ObjectType::TRIANGLE;
        checkCudaErrors(cudaMallocManaged(&object->triangle, sizeof(TriangleParams)));
        parseTriangle(object->triangle);
    } else if (!strcmp(token, "TriangleMesh")) {
        *object_type = ObjectType::MESH;
        checkCudaErrors(cudaMallocManaged(&object->mesh, sizeof(MeshParams)));
        parseMesh(object->mesh);
    } else if (!strcmp(token, "Transform")) {
        *object_type = ObjectType::TRANSFORM;
        checkCudaErrors(cudaMallocManaged(&object->transform, sizeof(TransformParams)));
        parseTransform(object->transform);
    } else if (!strcmp(token, "BezierCurve")) {
        *object_type = ObjectType::CURVE;
        checkCudaErrors(cudaMallocManaged(&object->curve, sizeof(CurveParams)));
        parseBezierCurve(object->curve);
    } else if (!strcmp(token, "BsplineCurve")) {
        *object_type = ObjectType::CURVE;
        checkCudaErrors(cudaMallocManaged(&object->curve, sizeof(CurveParams)));
        parseBsplineCurve(object->curve);
    } else if (!strcmp(token, "RevSurface")) {
        *object_type = ObjectType::REVSURFACE;
        checkCudaErrors(cudaMallocManaged(&object->revsurface, sizeof(RevsurfaceParams)));
        parseRevSurface(object->revsurface);
    } else {
        printf("Unknown token in parseObject: '%s'\n", token);
        exit(0);
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parseGroup(GroupParams *group_params) {
    if (debug)
        printf("SceneParser::parseGroup(0x%lx)\n", group_params);
    //
    // each group starts with an integer that specifies
    // the number of objects in the group
    //
    // the material index sets the material of all objects which follow,
    // until the next material index (scoping for the materials is very
    // simple, and essentially ignores any tree hierarchy)
    //
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    // printf("\t%s\n", token);

    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numObjects"));
    // printf("\t%s\n", token);

    group_params->num_objects = readInt();
    checkCudaErrors(cudaMallocManaged(
        &group_params->objects, group_params->num_objects * sizeof(ObjectParamsPointer)));
    // printf("\tgroup_params->objects:\t0x%lx\n", group_params->objects);
    checkCudaErrors(cudaMallocManaged(&group_params->object_types,
                                      group_params->num_objects * sizeof(ObjectType)));
    // printf("\tgroup_params->object_types:\t0x%lx\n", group_params->object_types);

    // read in the objects
    int count = 0;
    while (count < group_params->num_objects) {
        getToken(token);
        // printf("\t%s\n", token);
        if (!strcmp(token, "MaterialIndex")) {
            // change the current material
            current_material = readInt();
            assert(0 <= current_material &&
                   current_material <= materials_params->num_materials);
        } else {
            // printf("\t&group_params->objects[count]:\t0x%lx\n",
            //&group_params->objects[count]);
            // printf("\t&group_params->object_types[count]:\t0x%lx\n",
            //&group_params->object_types[count]);
            parseObject(token, &group_params->objects[count],
                        &group_params->object_types[count]);

            count++;
        }
        // printf("\n");
    }
    getToken(token);
    assert(!strcmp(token, "}"));
}

// ====================================================================
// ====================================================================

void SceneParser::parseSphere(SphereParams *sphere_params) {
    if (debug)
        printf("SceneParser::parseSphere(0x%lx)\n", sphere_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "center"));
    sphere_params->center = readVector3f();

    getToken(token);
    assert(!strcmp(token, "radius"));
    sphere_params->radius = readFloat();

    getToken(token);
    assert(!strcmp(token, "}"));

    assert(current_material != -1);
    sphere_params->material_id = current_material;
}

void SceneParser::parsePlane(PlaneParams *plane_params) {
    if (debug)
        printf("SceneParser::parsePlane(0x%lx)\n", plane_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    plane_params->normal = -Vector3f::FORWARD;
    plane_params->d = 0.f;
    plane_params->texture_origin = Vector3f::ZERO;
    plane_params->texture_x = Vector3f::RIGHT;
    plane_params->texture_y = Vector3f::UP;

    while (true) {
        getToken(token);

        if (!strcmp(token, "normal")) {
            plane_params->normal = readVector3f();
        } else if (!strcmp(token, "offset")) {
            plane_params->d = readFloat();
        } else if (!strcmp(token, "texture_origin")) {
            plane_params->texture_origin = readVector3f();
        } else if (!strcmp(token, "texture_x")) {
            plane_params->texture_x = readVector3f();
        } else if (!strcmp(token, "texture_y")) {
            plane_params->texture_y = readVector3f();
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }

    assert(current_material != -1);
    plane_params->material_id = current_material;
}

void SceneParser::parseTriangle(TriangleParams *triangle_params) {
    if (debug)
        printf("SceneParser::parseTriangle(0x%lx)\n", triangle_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "vertex0"));
    triangle_params->vertices[0] = readVector3f();

    getToken(token);
    assert(!strcmp(token, "vertex1"));
    triangle_params->vertices[1] = readVector3f();

    getToken(token);
    assert(!strcmp(token, "vertex2"));
    triangle_params->vertices[2] = readVector3f();

    getToken(token);
    assert(!strcmp(token, "}"));

    assert(current_material != -1);
    triangle_params->material_id = current_material;
}

void SceneParser::parseMesh(MeshParams *mesh_params) {
    if (debug)
        printf("SceneParser::parseMesh(0x%lx)\n", mesh_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];

    // get the filename
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "obj_file"));

    getToken(filename);

    getToken(token);
    assert(!strcmp(token, "}"));

    const char *ext = &filename[strlen(filename) - 4];
    assert(!strcmp(ext, ".obj"));

    // parse obj model
    // @TODO Optional: Use tiny obj loader to replace this simple one.
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    std::string tok;
    int texID;

    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<dim3> triangles;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vector3f vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            vertices.push_back(vec);
        } else if (tok == fTok) {
            dim3 trig;
            if (line.find('/') != std::string::npos) {
                std::replace(line.begin(), line.end(), '/', ' ');
                std::stringstream facess(line);
                facess >> tok;
                facess >> trig.x >> texID;
                facess >> trig.y >> texID;
                facess >> trig.z >> texID;
            } else {
                ss >> trig.x;
                ss >> trig.y;
                ss >> trig.z;
            }
            trig.x -= 1;
            trig.y -= 1;
            trig.z -= 1;
            triangles.push_back(trig);
        } else if (tok == texTok) {
            Vector2f texcoord;
            ss >> texcoord[0];
            ss >> texcoord[1];
        }
    }

    f.close();

    mesh_params->min = Vector3f(INFINITY, INFINITY, INFINITY);
    mesh_params->max = Vector3f(-INFINITY, -INFINITY, -INFINITY);
    int num_vertices = vertices.size();
    for (int i = 0; i < num_vertices; i++) {
        auto &v = vertices[i];
        if (v[0] < mesh_params->min[0]) {
            mesh_params->min[0] = v[0];
        }
        if (v[1] < mesh_params->min[1]) {
            mesh_params->min[1] = v[1];
        }
        if (v[2] < mesh_params->min[2]) {
            mesh_params->min[2] = v[2];
        }
        if (v[0] > mesh_params->max[0]) {
            mesh_params->max[0] = v[0];
        }
        if (v[1] > mesh_params->max[1]) {
            mesh_params->max[1] = v[1];
        }
        if (v[2] > mesh_params->max[2]) {
            mesh_params->max[2] = v[2];
        }
    }

    mesh_params->num_triangles = triangles.size();
    checkCudaErrors(cudaMallocManaged(&mesh_params->triangle_vertices,
                                      3 * mesh_params->num_triangles * sizeof(Vector3f)));
    for (int i = 0; i < mesh_params->num_triangles; i++) {
        mesh_params->triangle_vertices[3 * i] = vertices[triangles[i].x];
        mesh_params->triangle_vertices[3 * i + 1] = vertices[triangles[i].y];
        mesh_params->triangle_vertices[3 * i + 2] = vertices[triangles[i].z];
    }

    assert(current_material != -1);
    mesh_params->material_id = current_material;
}

void SceneParser::parseBezierCurve(CurveParams *curve_params) {
    if (debug)
        printf("SceneParser::parseBezierCurve(0x%lx)\n", curve_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "controls"));

    thrust::host_vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            Vector3f point = readVector3f();
            point.x() = fabs(point.x());
            controls.push_back(point);
            getToken(token);
            assert(!strcmp(token, "]"));
        } else if (!strcmp(token, "}")) {
            break;
        } else {
            printf("Incorrect format for BezierCurve!\n");
            exit(0);
        }
    }

    curve_params->num_controls = controls.size();
    checkCudaErrors(cudaMallocManaged(&curve_params->controls,
                                      curve_params->num_controls * sizeof(Vector3f)));
    if (debug)
        printf("curve_params->controls:\t0x%lx\n", curve_params->controls);
    thrust::copy(controls.begin(), controls.end(), curve_params->controls);
    curve_params->type = CurveParams::Type::Bezier;
}

void SceneParser::parseBsplineCurve(CurveParams *curve_params) {
    if (debug)
        printf("SceneParser::parseBsplineCurve(0x%lx)\n", curve_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "controls"));

    thrust::host_vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            Vector3f point = readVector3f();
            point.x() = fabs(point.x());
            controls.push_back(point);
            getToken(token);
            assert(!strcmp(token, "]"));
        } else if (!strcmp(token, "}")) {
            break;
        } else {
            printf("Incorrect format for BsplineCurve!\n");
            exit(0);
        }
    }

    curve_params->num_controls = controls.size();
    checkCudaErrors(cudaMallocManaged(&curve_params->controls,
                                      curve_params->num_controls * sizeof(Vector3f)));
    if (debug)
        printf("curve_params->controls:\t0x%lx\n", curve_params->controls);
    thrust::copy(controls.begin(), controls.end(), curve_params->controls);
    curve_params->type = CurveParams::Type::BSpline;
}

void SceneParser::parseRevSurface(RevsurfaceParams *revsurface_params) {
    if (debug)
        printf("SceneParser::parseRevSurface(0x%lx)\n", revsurface_params);
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "profile"));

    getToken(token);
    if (!strcmp(token, "BezierCurve")) {
        checkCudaErrors(
            cudaMallocManaged(&revsurface_params->curve, sizeof(CurveParams)));
        if (debug)
            printf("revsurface_params->curve:\t0x%lx\n", revsurface_params->curve);
        parseBezierCurve(revsurface_params->curve);
    } else if (!strcmp(token, "BsplineCurve")) {
        checkCudaErrors(
            cudaMallocManaged(&revsurface_params->curve, sizeof(CurveParams)));
        if (debug)
            printf("revsurface_params->curve:\t0x%lx\n", revsurface_params->curve);
        parseBsplineCurve(revsurface_params->curve);
    } else {
        printf("Unknown profile type in parseRevSurface: '%s'\n", token);
        exit(0);
    }

    getToken(token);
    assert(!strcmp(token, "}"));

    assert(current_material != -1);
    revsurface_params->material_id = current_material;
}

void SceneParser::parseTransform(TransformParams *transform_params) {
    if (debug)
        printf("SceneParser::parseTransform(0x%lx)\n", transform_params);
    char token[MAX_PARSER_TOKEN_LENGTH];

    auto &matrix = transform_params->matrix;
    matrix = Matrix4f::identity();

    getToken(token);
    assert(!strcmp(token, "{"));

    // read in transformations:
    // apply to the LEFT side of the current matrix (so the first
    // transform in the list is the last applied to the object)
    getToken(token);

    while (true) {
        if (!strcmp(token, "Scale")) {
            Vector3f s = readVector3f();
            matrix = matrix * Matrix4f::scaling(s[0], s[1], s[2]);
        } else if (!strcmp(token, "UniformScale")) {
            float s = readFloat();
            matrix = matrix * Matrix4f::uniformScaling(s);
        } else if (!strcmp(token, "Translate")) {
            matrix = matrix * Matrix4f::translation(readVector3f());
        } else if (!strcmp(token, "XRotate")) {
            matrix = matrix * Matrix4f::rotateX(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "YRotate")) {
            matrix = matrix * Matrix4f::rotateY(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "ZRotate")) {
            matrix = matrix * Matrix4f::rotateZ(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "Rotate")) {
            getToken(token);
            assert(!strcmp(token, "{"));
            Vector3f axis = readVector3f();
            float degrees = readFloat();
            float radians = DegreesToRadians(degrees);
            matrix = matrix * Matrix4f::rotation(axis, radians);
            getToken(token);
            assert(!strcmp(token, "}"));
        } else if (!strcmp(token, "Matrix4f")) {
            Matrix4f matrix2 = Matrix4f::identity();
            getToken(token);
            assert(!strcmp(token, "{"));
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    float v = readFloat();
                    matrix2(i, j) = v;
                }
            }
            getToken(token);
            assert(!strcmp(token, "}"));
            matrix = matrix2 * matrix;
        } else {
            // otherwise this must be an object,
            // and there are no more transformations
            parseObject(token, &transform_params->object, &transform_params->object_type);
            break;
        }
        getToken(token);
    }

    getToken(token);
    assert(!strcmp(token, "}"));
}

void SceneParser::freeBaseGroupParams() {
    if (debug)
        printf("SceneParser::freeBaseGroupParams()\n");
    if (base_group_params == nullptr) {
        return;
    }

    std::stack<GroupParams *> S_group;
    S_group.push(base_group_params);

    while (!S_group.empty()) {
        GroupParams *group = S_group.top();
        S_group.pop();
        if (debug)
            printf("Group*:\t0x%lx\n", group);

        for (int i = 0; i < group->num_objects; i++) {
            ObjectParamsPointer &object = group->objects[i];
            switch (group->object_types[i]) {
            case ObjectType::TRIANGLE:
                if (debug)
                    printf("Triangle*:\t0x%lx\n", object.triangle);
                checkCudaErrors(cudaFree(object.triangle));
                break;
            case ObjectType::SPHERE:
                if (debug)
                    printf("Sphere*:\t0x%lx\n", object.sphere);
                checkCudaErrors(cudaFree(object.sphere));
                break;
            case ObjectType::PLANE:
                if (debug)
                    printf("Plane*:\t0x%lx\n", object.plane);
                checkCudaErrors(cudaFree(object.plane));
                break;
            case ObjectType::MESH:
                if (debug)
                    printf("MESH*:\t0x%lx\n", object.mesh);
                checkCudaErrors(cudaFree(object.mesh->triangle_vertices));
                checkCudaErrors(cudaFree(object.mesh));
                break;
            case ObjectType::REVSURFACE:
                if (debug)
                    printf("RevSurface*:\t0x%lx\n", object.revsurface);
                checkCudaErrors(cudaFree(object.revsurface->curve->controls));
                checkCudaErrors(cudaFree(object.revsurface->curve));
                checkCudaErrors(cudaFree(object.revsurface));
                break;
            case ObjectType::TRANSFORM: {
                TransformParams *transform = object.transform;

                while (transform->object_type == ObjectType::TRANSFORM) {
                    TransformParams *prev = transform;
                    transform = prev->object.transform;
                    if (debug)
                        printf("Transform*:\t0x%lx\n", prev);
                    checkCudaErrors(cudaFree(prev));
                }

                switch (transform->object_type) {
                case ObjectType::TRIANGLE:
                    if (debug)
                        printf("Triangle*:\t0x%lx\n", transform->object.triangle);
                    checkCudaErrors(cudaFree(transform->object.triangle));
                    break;
                case ObjectType::SPHERE:
                    if (debug)
                        printf("Sphere*:\t0x%lx\n", transform->object.sphere);
                    checkCudaErrors(cudaFree(transform->object.sphere));
                    break;
                case ObjectType::PLANE:
                    if (debug)
                        printf("Plane*:\t0x%lx\n", transform->object.plane);
                    checkCudaErrors(cudaFree(transform->object.plane));
                    break;
                case ObjectType::MESH:
                    if (debug)
                        printf("Mesh*:\t0x%lx\n", transform->object.mesh);
                    checkCudaErrors(cudaFree(transform->object.mesh->triangle_vertices));
                    checkCudaErrors(cudaFree(transform->object.mesh));
                    break;
                case ObjectType::REVSURFACE:
                    if (debug)
                        printf("RevSurface*:\t0x%lx\n", transform->object.revsurface);
                    checkCudaErrors(
                        cudaFree(transform->object.revsurface->curve->controls));
                    checkCudaErrors(cudaFree(transform->object.revsurface->curve));
                    checkCudaErrors(cudaFree(transform->object.revsurface));
                    break;
                case ObjectType::GROUP:
                    if (debug)
                        printf("Group*:\t0x%lx\n", transform->object.group);
                    S_group.push(transform->object.group);
                    break;
                default:
                    std::cerr << "SceneParser::freeBaseGroupParams\tUnknown object type"
                              << std::endl;
                    assert(false);
                    break;
                }

                if (debug)
                    printf("Transform*:\t0x%lx\n", transform);
                checkCudaErrors(cudaFree(transform));

                break;
            }
            case ObjectType::GROUP:
                if (debug)
                    printf("Group*:\t0x%lx\n", object.group);
                S_group.push(object.group);
                break;
            default:
                std::cerr << "SceneParser::freeBaseGroupParams\tUnknown object type"
                          << std::endl;
                assert(false);
                break;
            }
        }

        checkCudaErrors(cudaFree(group->objects));
        checkCudaErrors(cudaFree(group->object_types));
        checkCudaErrors(cudaFree(group));
    }
}

// ====================================================================
// ====================================================================

int SceneParser::getToken(char token[MAX_PARSER_TOKEN_LENGTH]) {
    // for simplicity, tokens must be separated by whitespace
    assert(file != nullptr);
    int success = fscanf(file, "%s ", token);
    if (success == EOF) {
        token[0] = '\0';
        return 0;
    }
    return 1;
}

Vector3f SceneParser::readVector3f() {
    float x, y, z;
    int count = fscanf(file, "%f %f %f", &x, &y, &z);
    if (count != 3) {
        printf("Error trying to read 3 floats to make a Vector3f\n");
        assert(0);
    }
    // printf("\t%f, %f, %f\n", x, y, z);
    return Vector3f(x, y, z);
}

float SceneParser::readFloat() {
    float answer;
    int count = fscanf(file, "%f", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert(0);
    }
    // printf("\t%f\n", answer);
    return answer;
}

double SceneParser::readDouble() {
    double answer;
    int count = fscanf(file, "%lf", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert(0);
    }
    // printf("\t%lf\n", answer);
    return answer;
}

int SceneParser::readInt() {
    int answer;
    int count = fscanf(file, "%d", &answer);
    if (count != 1) {
        printf("Error trying to read 1 int\n");
        assert(0);
    }
    // printf("\t%d\n", answer);
    return answer;
}
