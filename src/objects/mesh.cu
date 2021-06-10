#include "mesh.h"

__device__ bool Mesh::intersect(const Ray &ray, Hit &hit, float t_min, curandState *rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }

    // @TODO Optional: Change this brute force method into a faster one.
    bool result = false;
    for (int i = 0; i < num_faces; ++i) {
        result |= faces[i]->intersect(ray, hit, t_min, rand_state);
    }
    return result;
}

Mesh::Mesh(const char *filename, Material *material) : Object3D(material) {
    // Optional: Use tiny obj loader to replace this simple one.
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

    std::vector<Vector3f> v;
    std::vector<dim3> t;
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

    num_vertices = v.size();
    num_faces = t.size();

    checkCudaErrors(cudaMallocManaged(&vertices, num_vertices * sizeof(Vector3f)));
    checkCudaErrors(cudaMallocManaged(&faces, num_faces * sizeof(Triangle *)));
    for (int i = 0; i < num_vertices; i++) {
        vertices[i] = v[i];
    }
    for (int i = 0; i < num_faces; i++) {
        faces[i] = new Triangle(vertices[t[i].x], vertices[t[i].y], vertices[t[i].z], material);
    }

    Vector3f min = vertices[0];
    Vector3f max = vertices[0];

    for (int i = 1; i < num_vertices; i++) {
        auto &p = vertices[i];
        for (int j = 0; j < 3; j++) {
            if (p[j] < min[j]) {
                min[j] = p[j];
            }
            if (p[j] > max[j]) {
                max[j] = p[j];
            }
        }
    }

    pBox = new BoundingBox(min, max);
}

Mesh::~Mesh() {
    delete pBox;
    checkCudaErrors(cudaFree(vertices));
    checkCudaErrors(cudaFree(faces));
}
