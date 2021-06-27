#include "mesh.h"

__device__ Mesh::Mesh(Vector3f *triangle_vertices, int num_triangles, Vector3f min,
                      Vector3f max, Material *material, float curve_step)
    : Object3D(material), num_triangles(num_triangles), curve_step(curve_step) {
    triangles = new Triangle *[num_triangles];
    for (int i = 0; i < num_triangles; i++) {
        triangles[i] =
            new Triangle(triangle_vertices[3 * i], triangle_vertices[3 * i + 1],
                         triangle_vertices[3 * i + 2], material, i);
    }

    pBox = new BoundingBox(min, max);

    pRoot = new Group(num_triangles);
    static_cast<Group *>(pRoot)->num_objects = num_triangles;
    memcpy(static_cast<Group *>(pRoot)->objects, triangles,
           num_triangles * sizeof(Triangle *));

    if (max_BSP_depth == 0 || num_triangles <= max_BSP_leaf_size) {
        return;
    }

    Space **S[max_BSP_depth + 1];
    Axis axes[max_BSP_depth + 1];
    int depths[max_BSP_depth + 1];
    int p_S = 0;
    S[p_S] = &pRoot;
    axes[p_S] = Axis::X;
    depths[p_S++] = 0;
    while (p_S > 0) {
        p_S--;
        Group *pGroup = static_cast<Group *>(*S[p_S]);
        Space **ptrFromParent = S[p_S];
        Axis axis = axes[p_S];
        int depth = depths[p_S];

        Triangle **pTris = (Triangle **)pGroup->objects;
        int num_tris = pGroup->num_objects;

        float key;
        int l_prev = 0;
        int r_prev = num_tris - 1;
        int mid = num_tris / 2;
        while (l_prev < r_prev) {
            int l = l_prev;
            int r = r_prev;
            Triangle *pivot = pTris[l];
            key = pTris[l]->getCenter()[axis];
            while (l < r) {
                while (l < r && pTris[r]->getCenter()[axis] >= key) {
                    r--;
                }
                pTris[l] = pTris[r];
                while (l < r && pTris[l]->getCenter()[axis] <= key) {
                    l++;
                }
                pTris[r] = pTris[l];
            }
            pTris[l] = pivot;
            if (l <= mid) {
                l_prev = l + 1;
            } 
            if (l >= mid - 1) {
                r_prev = r - 1;
            }
        }

        Group *pRChild = new Group(num_tris);
        pRChild->num_objects = num_tris - mid;
        memcpy(pRChild->objects, &pTris[mid], pRChild->num_objects * sizeof(Triangle *));
        for (int i = 0; i < mid; i++) {
            if (pTris[i]->intersect_plane(axis, key)) {
                pRChild->addObject(pTris[i]);
            }
        }

        if (pRChild->num_objects == num_tris) {
            delete pRChild;
            continue;
        }

        Group *pLChild = pGroup;
        pLChild->num_objects = mid;
        for (int i = mid; i < num_tris; i++) {
            if (pTris[i]->intersect_plane(axis, key)) {
                pLChild->addObject(pTris[i]);
            }
        }

        // pLChild->shrink_to_fit();
        // pRChild->shrink_to_fit();

        BSPNode *pBSPNode = new BSPNode(axis, key);
        pBSPNode->lChild = pLChild;
        pBSPNode->rChild = pRChild;

        *ptrFromParent = pBSPNode;

        depth++;
        axis = Axis(axis + (axis == 2 ? -2 : 1));
        if (depth < max_BSP_depth && pLChild->num_objects > max_BSP_leaf_size) {
            S[p_S] = &pBSPNode->lChild;
            axes[p_S] = axis;
            depths[p_S++] = depth;
        }
        if (depth < max_BSP_depth && pRChild->num_objects > max_BSP_leaf_size) {
            S[p_S] = &pBSPNode->rChild;
            axes[p_S] = axis;
            depths[p_S++] = depth;
        }
    }
}

__device__ Mesh::~Mesh() {
    delete pBox;
    for (int i = 0; i < num_triangles; i++) {
        delete triangles[i];
    }
    delete[] triangles;
    delete pRoot;
}

__device__ bool Mesh::intersect(const Ray &ray, Hit &hit, float t_min,
                                RandState &rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }
    return pRoot->intersect(ray, hit, t_min, rand_state);
}

__device__ bool Mesh::intersect(const Ray &ray, float t_min, float t_max,
                                RandState &rand_state) {
    if (!pBox->intersect(ray, t_min, t_max)) {
        return false;
    }
    return pRoot->intersect(ray, t_min, t_max, rand_state);
}

__device__ bool Mesh::intersect_rev(const Ray &ray, Hit &hit, float t_min,
                                    RandState &rand_state) {
    if (!pBox->intersect(ray, t_min)) {
        return false;
    }

    bool res = pRoot->intersect(ray, hit, t_min, rand_state);
    if (res) {
        int u_level = hit.id / (angle_steps * 2);
        float u = float(u_level) * curve_step;

        Vector3f point = ray.pointAtParameter(hit.getT());
        float r = sqrt(point.x() * point.x() + point.z() * point.z());
        float sinv = point.z() / r;
        float cosv = point.x() / r;
        float v = sinv / (1.f + cosv);
        hit.set(hit.getT(), u, v);
    }

    return res;
}

__device__ bool Mesh::intersect_rev(const Ray &ray, float t_min, float t_max,
                                    RandState &rand_state) {
    if (!pBox->intersect(ray, t_min, t_max)) {
        return false;
    }

    return pRoot->intersect(ray, t_min, t_max, rand_state);
}