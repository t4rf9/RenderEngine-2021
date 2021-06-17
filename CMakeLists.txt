CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(main LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75-real)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

ADD_SUBDIRECTORY(deps/vecmath)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

ADD_EXECUTABLE(main
        src/device/camera/camera.h src/device/camera/camera.cu
        src/device/camera/perspective_camera.cu src/device/camera/perspective_camera.h
        src/device/camera/cameras.h

        src/device/lights/lights.h
        src/device/lights/light.h src/device/lights/light.cu
        src/device/lights/directional_light.h src/device/lights/directional_light.cu
        src/device/lights/point_light.h src/device/lights/point_light.cu
        src/device/lights/disk_light.h src/device/lights/disk_light.cu

        src/device/material.h src/device/material.cu

        src/device/objects/group.h src/device/objects/group.cu
        src/device/objects/mesh.h src/device/objects/mesh.cu
        src/device/objects/object3d.h src/device/objects/object3d.cu
        src/device/objects/plane.h src/device/objects/plane.cu
        src/device/objects/revsurface.h src/device/objects/revsurface.cu
        src/device/objects/curve.h src/device/objects/curve.cu
        src/device/objects/BezierCurve.h src/device/objects/BezierCurve.cu
        src/device/objects/BsplineCurve.h src/device/objects/BsplineCurve.cu
        src/device/objects/sphere.h src/device/objects/sphere.cu
        src/device/objects/transform.h src/device/objects/transform.cu
        src/device/objects/triangle.h src/device/objects/triangle.cu
        src/device/objects/BoundingBox.h src/device/objects/BoundingBox.cu
        src/device/objects/BoundingObject.h src/device/objects/BoundingObject.cu
        src/device/hit.h src/device/hit.cu
        src/device/ray.h src/device/ray.cu

        src/scene/scene_parser.h src/scene/scene_parser.cu
        src/scene/scene.h src/scene/scene.cu
        
        src/scene/parameters/parameters.h
        src/scene/parameters/camera_params.h
        src/scene/parameters/lights_params.h
        src/scene/parameters/light_params.h
        src/scene/parameters/materials_params.h
        src/scene/parameters/material_params.h

        src/scene/parameters/objects/curve_params.h
        src/scene/parameters/objects/group_params.h
        src/scene/parameters/objects/mesh_params.h
        src/scene/parameters/objects/plane_params.h
        src/scene/parameters/objects/revsurface_params.h
        src/scene/parameters/objects/sphere_params.h
        src/scene/parameters/objects/transform_params.h
        src/scene/parameters/objects/triangle_params.h
        src/scene/parameters/objects/object_type.h
        src/scene/parameters/objects/object_params_pointer.h

        src/image.h src/image.cu
        src/cuda_error.h src/cuda_error.cu
        src/render.h src/render.cu
        src/create_scene.h src/create_scene.cu
        src/destroy_scene.h src/destroy_scene.cu

        src/main.cu
        )

ADD_EXECUTABLE(test
        src/device/camera/camera.h src/device/camera/camera.cu
        src/device/camera/perspective_camera.cu src/device/camera/perspective_camera.h
        src/device/camera/cameras.h

        src/device/lights/lights.h
        src/device/lights/light.h src/device/lights/light.cu
        src/device/lights/directional_light.h src/device/lights/directional_light.cu
        src/device/lights/point_light.h src/device/lights/point_light.cu
        src/device/lights/disk_light.h src/device/lights/disk_light.cu

        src/device/material.h src/device/material.cu

        src/device/objects/group.h src/device/objects/group.cu
        src/device/objects/mesh.h src/device/objects/mesh.cu
        src/device/objects/object3d.h src/device/objects/object3d.cu
        src/device/objects/plane.h src/device/objects/plane.cu
        src/device/objects/revsurface.h src/device/objects/revsurface.cu
        src/device/objects/curve.h src/device/objects/curve.cu
        src/device/objects/BezierCurve.h src/device/objects/BezierCurve.cu
        src/device/objects/BsplineCurve.h src/device/objects/BsplineCurve.cu
        src/device/objects/sphere.h src/device/objects/sphere.cu
        src/device/objects/transform.h src/device/objects/transform.cu
        src/device/objects/triangle.h src/device/objects/triangle.cu
        src/device/objects/BoundingBox.h src/device/objects/BoundingBox.cu
        src/device/objects/BoundingObject.h src/device/objects/BoundingObject.cu
        src/device/hit.h src/device/hit.cu
        src/device/ray.h src/device/ray.cu

        src/scene/scene_parser.h src/scene/scene_parser.cu
        src/scene/scene.h src/scene/scene.cu

        src/scene/parameters/parameters.h
        src/scene/parameters/camera_params.h
        src/scene/parameters/lights_params.h
        src/scene/parameters/light_params.h
        src/scene/parameters/materials_params.h
        src/scene/parameters/material_params.h

        src/scene/parameters/objects/curve_params.h
        src/scene/parameters/objects/group_params.h
        src/scene/parameters/objects/mesh_params.h
        src/scene/parameters/objects/plane_params.h
        src/scene/parameters/objects/revsurface_params.h
        src/scene/parameters/objects/sphere_params.h
        src/scene/parameters/objects/transform_params.h
        src/scene/parameters/objects/triangle_params.h
        src/scene/parameters/objects/object_type.h
        src/scene/parameters/objects/object_params_pointer.h

        src/image.h src/image.cu
        src/cuda_error.h src/cuda_error.cu
        src/render.h src/render.cu
        src/create_scene.h src/create_scene.cu
        src/destroy_scene.h src/destroy_scene.cu

        src/test.cu
        )

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET vecmath PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(main PUBLIC -G -g)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(test ${CUDA_LIBRARIES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC src)

TARGET_LINK_LIBRARIES(test vecmath)
TARGET_INCLUDE_DIRECTORIES(test PUBLIC src)
