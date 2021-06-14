#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "BoundingBox.h"
#include "Vector2f.h"
#include "Vector3f.h"
#include "object3d.h"
#include "triangle.h"

class Mesh : public Object3D {
public:
    Mesh(const char *filename, Material *m);

    ~Mesh();

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front face
        int x[3]{};
    };

    std::vector<Vector3f> v;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;
    bool intersect(const Ray &r, Hit &h, float tmin, uint_fast32_t &rand) override;

private:
    // Normal can be used for light estimation
    void computeNormal();

    BoundingBox *pBox;
};
