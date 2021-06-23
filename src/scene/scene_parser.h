#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stack>

#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <vecmath.h>

#include "cuda_error.h"

#include "define.h"
#include "scene/parameters/parameters.h"
#include "image.h"

#define MAX_PARSER_TOKEN_LENGTH 1024

class SceneParser {
public:
    SceneParser() = delete;
    SceneParser(const char *filename);

    ~SceneParser();

    inline Vector3f getBackgroundColor() const { return background_color; }

    inline Vector3f getEnvironmentColor() const { return environment_color; }

    inline CameraParams *getCameraParams() const { return camera_params; }

    inline LightsParams *getLightsParams() const { return lights_params; }

    inline MaterialsParams *getMaterialsParams() const { return materials_params; }

    inline GroupParams *getBaseGroupParams() const { return base_group_params; }

private:
    void parseFile();

    void parsePerspectiveCamera();

    void parseBackground();

    void parseLights();
    void parsePointLight(LightParams *light_param);
    void parseDirectionalLight(LightParams *light_param);
    void parseDiskLight(LightParams *light_param);

    void parseMaterials();
    void parsePhongMaterial(MaterialParams *material_params);

    void parseObject(char token[MAX_PARSER_TOKEN_LENGTH], ObjectParamsPointer *object,
                     ObjectType *object_type);
    void parseGroup(GroupParams *group_params);
    void parseSphere(SphereParams *sphere_params);
    void parsePlane(PlaneParams *plane_params);
    void parseTriangle(TriangleParams *triangle_params);
    void parseMesh(MeshParams *mesh_params);
    void parseTransform(TransformParams *transform_params);
    void parseBezierCurve(CurveParams *curve_params);
    void parseBsplineCurve(CurveParams *curve_params);
    void parseRevSurface(RevsurfaceParams *revsurface_params);

    void freeBaseGroupParams();

    int getToken(char token[MAX_PARSER_TOKEN_LENGTH]);

    Vector3f readVector3f();

    float readFloat();
    double readDouble();
    int readInt();

    FILE *file;

    Vector3f background_color = Vector3f(0.5f, 0.5f, 0.5f);
    Vector3f environment_color = Vector3f::ZERO;

    CameraParams *camera_params = nullptr;
    LightsParams *lights_params = nullptr;
    MaterialsParams *materials_params = nullptr;
    GroupParams *base_group_params = nullptr;

    int current_material = -1;
};

#endif // SCENE_PARSER_H
