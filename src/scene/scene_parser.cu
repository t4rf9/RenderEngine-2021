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

    checkCudaErrors(cudaMallocManaged(&lights_params, sizeof(LightsParams)));
    lights_params->num_lights = 0;

    checkCudaErrors(cudaMallocManaged(&materials_params, sizeof(MaterialsParams)));
    materials_params->num_materials = 0;

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

    for (int i = 0; i < lights_params->num_lights; i++) {
        checkCudaErrors(cudaFree(lights_params->lights));
    }
    checkCudaErrors(cudaFree(lights_params));

    for (int i = 0; i < materials_params->num_materials; i++) {
        checkCudaErrors(cudaFree(materials_params->materials));
    }
    checkCudaErrors(cudaFree(materials_params));

    freeBaseGroupParams();
}

// ====================================================================
// ====================================================================

void SceneParser::parseFile() {
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
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the camera parameters
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "center"));
    camera_params->pos = readVector3f();

    getToken(token);
    assert(!strcmp(token, "direction"));
    camera_params->direction = readVector3f();

    getToken(token);
    assert(!strcmp(token, "up"));
    camera_params->up = readVector3f();

    getToken(token);
    assert(!strcmp(token, "angle"));
    float angle_degrees = readFloat();
    camera_params->angle = DegreesToRadians(angle_degrees);

    getToken(token);
    assert(!strcmp(token, "width"));
    camera_params->width = readInt();

    getToken(token);
    assert(!strcmp(token, "height"));
    camera_params->height = readInt();

    getToken(token);
    assert(!strcmp(token, "}"));

    camera_params->type = CameraParams::Type::PerspectiveCamera;
}

void SceneParser::parseBackground() {
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
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    // read in the number of lights
    getToken(token);
    assert(!strcmp(token, "numLights"));
    lights_params->num_lights = readInt();
    checkCudaErrors(cudaMallocManaged(&lights_params->lights,
                                      lights_params->num_lights * sizeof(LightParams)));

    // read in the objects
    int count = 0;
    while (count < lights_params->num_lights) {
        getToken(token);
        if (strcmp(token, "DirectionalLight") == 0) {
            parseDirectionalLight(&lights_params->lights[count]);
        } else if (strcmp(token, "PointLight") == 0) {
            parsePointLight(&lights_params->lights[count]);
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

    light_param->type = LightParams::Type::Directional;
}

void SceneParser::parsePointLight(LightParams *light_param) {
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

    light_param->type = LightParams::Type::Point;
}
// ====================================================================
// ====================================================================

void SceneParser::parseMaterials() {
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
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];
    filename[0] = 0;

    material_params->diffuseColor = Vector3f(1.f, 1.f, 1.f);
    material_params->specularColor = Vector3f(0.f, 0.f, 0.f);
    material_params->shininess = 0.f;
    material_params->reflect_coefficient = 0.f;
    material_params->refract_coefficient = 0.f;
    material_params->refractive_index = 1.f;

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
            // Optional: read in texture and draw it.
            getToken(filename);
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parseObject(char token[MAX_PARSER_TOKEN_LENGTH],
                              ObjectParamsPointer object, ObjectType *object_type) {
    if (!strcmp(token, "Group")) {
        *object_type = ObjectType::Group;
        parseGroup(object.group);
    } else if (!strcmp(token, "Sphere")) {
        *object_type = ObjectType::Sphere;
        parseSphere(object.sphere);
    } else if (!strcmp(token, "Plane")) {
        *object_type = ObjectType::Plane;
        parsePlane(object.plane);
    } else if (!strcmp(token, "Triangle")) {
        *object_type = ObjectType::Triangle;
        parseTriangle(object.triangle);
    } else if (!strcmp(token, "TriangleMesh")) {
        *object_type = ObjectType::Mesh;
        parseTriangleMesh(object.mesh);
    } else if (!strcmp(token, "Transform")) {
        *object_type = ObjectType::Transform;
        parseTransform(object.transform);
    } else if (!strcmp(token, "BezierCurve")) {
        *object_type = ObjectType::Curve;
        parseBezierCurve(object.curve);
    } else if (!strcmp(token, "BsplineCurve")) {
        *object_type = ObjectType::Curve;
        parseBsplineCurve(object.curve);
    } else if (!strcmp(token, "RevSurface")) {
        *object_type = ObjectType::RevSurface;
        parseRevSurface(object.revsurface);
    } else {
        printf("Unknown token in parseObject: '%s'\n", token);
        exit(0);
    }
}

// ====================================================================
// ====================================================================

void SceneParser::parseGroup(GroupParams *group_params) {
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

    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numObjects"));
    group_params->num_objects = readInt();
    checkCudaErrors(cudaMallocManaged(
        &group_params->objects, group_params->num_objects * sizeof(ObjectParamsPointer)));
    checkCudaErrors(cudaMallocManaged(&group_params->object_types,
                                      group_params->num_objects * sizeof(ObjectType)));

    // read in the objects
    int count = 0;
    while (count < group_params->num_objects) {
        getToken(token);
        if (!strcmp(token, "MaterialIndex")) {
            // change the current material
            current_material = readInt();
            assert(0 <= current_material &&
                   current_material <= materials_params->num_materials);
        } else {
            parseObject(token, group_params->objects[count],
                        &group_params->object_types[count]);

            count++;
        }
    }
    getToken(token);
    assert(!strcmp(token, "}"));
}

// ====================================================================
// ====================================================================

void SceneParser::parseSphere(SphereParams *sphere_params) {
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
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "normal"));
    plane_params->normal = readVector3f();

    getToken(token);
    assert(!strcmp(token, "offset"));
    plane_params->d = readFloat();

    getToken(token);
    assert(!strcmp(token, "}"));

    assert(current_material != -1);
    plane_params->material_id = current_material;
}

void SceneParser::parseTriangle(TriangleParams *triangle_params) {
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

void SceneParser::parseTriangleMesh(MeshParams *mesh_params) {
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

    thrust::host_vector<Vector3f> v;
    thrust::host_vector<dim3> t;
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
            v.push_back(vec);
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
            t.push_back(trig);
        } else if (tok == texTok) {
            Vector2f texcoord;
            ss >> texcoord[0];
            ss >> texcoord[1];
        }
    }

    f.close();

    mesh_params->num_vertices = v.size();
    mesh_params->num_faces = t.size();

    checkCudaErrors(cudaMallocManaged(&mesh_params->vertices,
                                      mesh_params->num_vertices * sizeof(Vector3f)));
    checkCudaErrors(
        cudaMallocManaged(&mesh_params->faces, mesh_params->num_faces * sizeof(dim3)));

    thrust::copy(v.begin(), v.end(), mesh_params->vertices);
    thrust::copy(t.begin(), t.end(), mesh_params->faces);

    assert(current_material != -1);
    mesh_params->material_id = current_material;
}

void SceneParser::parseBezierCurve(CurveParams *curve_params) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "controls"));

    thrust::host_vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            controls.push_back(readVector3f());
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
    thrust::copy(controls.begin(), controls.end(), curve_params->num_controls);
    curve_params->type = CurveParams::Type::Bezier;
}

void SceneParser::parseBsplineCurve(CurveParams *curve_params) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "controls"));

    thrust::host_vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            controls.push_back(readVector3f());
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
    thrust::copy(controls.begin(), controls.end(), curve_params->num_controls);
    curve_params->type = CurveParams::Type::BSpline;
}

void SceneParser::parseRevSurface(RevsurfaceParams *revsurface_params) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));

    getToken(token);
    assert(!strcmp(token, "profile"));

    getToken(token);
    if (!strcmp(token, "BezierCurve")) {
        parseBezierCurve(revsurface_params->curve);
    } else if (!strcmp(token, "BsplineCurve")) {
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
            parseObject(token, transform_params->object, &transform_params->object_type);
            break;
        }
        getToken(token);
    }

    getToken(token);
    assert(!strcmp(token, "}"));
}

void SceneParser::freeBaseGroupParams() {
    if (base_group_params == nullptr) {
        return;
    }

    std::stack<GroupParams *> S_group;
    S_group.push(base_group_params);

    while (!S_group.empty()) {
        GroupParams *group = S_group.top();
        S_group.pop();

        for (int i = 0; i < group->num_objects; i++) {
            ObjectParamsPointer &object = group->objects[i];
            switch (group->object_types[i]) {
            case ObjectType::Triangle:
                checkCudaErrors(cudaFree(object.triangle));
                break;
            case ObjectType::Sphere:
                checkCudaErrors(cudaFree(object.sphere));
                break;
            case ObjectType::Plane:
                checkCudaErrors(cudaFree(object.plane));
                break;
            case ObjectType::Mesh:
                checkCudaErrors(cudaFree(object.mesh->vertices));
                checkCudaErrors(cudaFree(object.mesh->faces));
                checkCudaErrors(cudaFree(object.mesh));
                break;
            case ObjectType::RevSurface:
                checkCudaErrors(cudaFree(object.revsurface->curve->controls));
                checkCudaErrors(cudaFree(object.revsurface->curve));
                checkCudaErrors(cudaFree(object.revsurface));
                break;
            case ObjectType::Transform:
                TransformParams *transform = object.transform;

                while (transform->object_type == ObjectType::Transform) {
                    TransformParams *prev = transform;
                    transform = prev->object.transform;
                    checkCudaErrors(cudaFree(prev));
                }

                switch (transform->object_type) {
                case ObjectType::Triangle:
                    checkCudaErrors(cudaFree(transform->object.triangle));
                    break;
                case ObjectType::Sphere:
                    checkCudaErrors(cudaFree(transform->object.sphere));
                    break;
                case ObjectType::Plane:
                    checkCudaErrors(cudaFree(transform->object.plane));
                    break;
                case ObjectType::Mesh:
                    checkCudaErrors(cudaFree(transform->object.mesh->vertices));
                    checkCudaErrors(cudaFree(transform->object.mesh->faces));
                    checkCudaErrors(cudaFree(transform->object.mesh));
                    break;
                case ObjectType::RevSurface:
                    checkCudaErrors(
                        cudaFree(transform->object.revsurface->curve->controls));
                    checkCudaErrors(cudaFree(transform->object.revsurface->curve));
                    checkCudaErrors(cudaFree(transform->object.revsurface));
                    break;
                case ObjectType::Group:
                    S_group.push(transform->object.group);
                    break;
                default:
                    std::cerr << "SceneParser::freeBaseGroupParams\tUnknown object type"
                              << std::endl;
                    assert(false);
                    break;
                }

                checkCudaErrors(cudaFree(transform));

                break;
            case ObjectType::Group:
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
    return Vector3f(x, y, z);
}

float SceneParser::readFloat() {
    float answer;
    int count = fscanf(file, "%f", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert(0);
    }
    return answer;
}

double SceneParser::readDouble() {
    double answer;
    int count = fscanf(file, "%lf", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert(0);
    }
    return answer;
}

int SceneParser::readInt() {
    int answer;
    int count = fscanf(file, "%d", &answer);
    if (count != 1) {
        printf("Error trying to read 1 int\n");
        assert(0);
    }
    return answer;
}
