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

        src/device/spaces/space.h src/device/spaces/space.cu
        src/device/spaces/group.h src/device/spaces/group.cu
        src/device/spaces/mesh.h src/device/spaces/mesh.cu
        src/device/spaces/object3d.h src/device/spaces/object3d.cu
        src/device/spaces/plane.h src/device/spaces/plane.cu
        src/device/spaces/revsurface.h src/device/spaces/revsurface.cu
        src/device/spaces/curve.h src/device/spaces/curve.cu
        src/device/spaces/BezierCurve.h src/device/spaces/BezierCurve.cu
        src/device/spaces/BsplineCurve.h src/device/spaces/BsplineCurve.cu
        src/device/spaces/sphere.h src/device/spaces/sphere.cu
        src/device/spaces/transform.h src/device/spaces/transform.cu
        src/device/spaces/triangle.h src/device/spaces/triangle.cu
        src/device/spaces/BoundingBox.h src/device/spaces/BoundingBox.cu
        src/device/spaces/BoundingObject.h src/device/spaces/BoundingObject.cu
        src/device/hit.h src/device/hit.cu
        src/device/ray.h src/device/ray.cu

        src/device/spaces/BSPNode.h src/device/spaces/BSPNode.cu

        src/scene/scene_parser.h src/scene/scene_parser.cu
        src/scene/scene.h src/scene/scene.cu
        
        src/scene/parameters/parameters.h
        src/scene/parameters/camera_params.h
        src/scene/parameters/lights_params.h
        src/scene/parameters/light_params.h
        src/scene/parameters/materials_params.h
        src/scene/parameters/material_params.h

        src/scene/parameters/spaces/curve_params.h
        src/scene/parameters/spaces/group_params.h
        src/scene/parameters/spaces/mesh_params.h
        src/scene/parameters/spaces/plane_params.h
        src/scene/parameters/spaces/revsurface_params.h
        src/scene/parameters/spaces/sphere_params.h
        src/scene/parameters/spaces/transform_params.h
        src/scene/parameters/spaces/triangle_params.h
        src/scene/parameters/spaces/object_type.h
        src/scene/parameters/spaces/object_params_pointer.h

        src/image.h src/image.cu
        src/cuda_error.h src/cuda_error.cu
        src/render.h src/render.cu
        src/create_scene.h src/create_scene.cu
        src/destroy_scene.h src/destroy_scene.cu

        src/main.cu
        )
       
#ADD_EXECUTABLE(test
#        src/device/camera/camera.h src/device/camera/camera.cu
#        src/device/camera/perspective_camera.cu src/device/camera/perspective_camera.h
#        src/device/camera/cameras.h
#
#        src/device/lights/lights.h
#        src/device/lights/light.h src/device/lights/light.cu
#        src/device/lights/directional_light.h src/device/lights/directional_light.cu
#        src/device/lights/point_light.h src/device/lights/point_light.cu
#        src/device/lights/disk_light.h src/device/lights/disk_light.cu
#
#        src/device/material.h src/device/material.cu
#
#        src/device/spaces/group.h src/device/spaces/group.cu
#        src/device/spaces/mesh.h src/device/spaces/mesh.cu
#        src/device/spaces/object3d.h src/device/spaces/object3d.cu
#        src/device/spaces/plane.h src/device/spaces/plane.cu
#        src/device/spaces/revsurface.h src/device/spaces/revsurface.cu
#        src/device/spaces/curve.h src/device/spaces/curve.cu
#        src/device/spaces/BezierCurve.h src/device/spaces/BezierCurve.cu
#        src/device/spaces/BsplineCurve.h src/device/spaces/BsplineCurve.cu
#        src/device/spaces/sphere.h src/device/spaces/sphere.cu
#        src/device/spaces/transform.h src/device/spaces/transform.cu
#        src/device/spaces/triangle.h src/device/spaces/triangle.cu
#        src/device/spaces/BoundingBox.h src/device/spaces/BoundingBox.cu
#        src/device/spaces/BoundingObject.h src/device/spaces/BoundingObject.cu
#        src/device/hit.h src/device/hit.cu
#        src/device/ray.h src/device/ray.cu
#
#        src/scene/scene_parser.h src/scene/scene_parser.cu
#        src/scene/scene.h src/scene/scene.cu
#
#        src/scene/parameters/parameters.h
#        src/scene/parameters/camera_params.h
#        src/scene/parameters/lights_params.h
#        src/scene/parameters/light_params.h
#        src/scene/parameters/materials_params.h
#        src/scene/parameters/material_params.h
#
#        src/scene/parameters/spaces/curve_params.h
#        src/scene/parameters/spaces/group_params.h
#        src/scene/parameters/spaces/mesh_params.h
#        src/scene/parameters/spaces/plane_params.h
#        src/scene/parameters/spaces/revsurface_params.h
#        src/scene/parameters/spaces/sphere_params.h
#        src/scene/parameters/spaces/transform_params.h
#        src/scene/parameters/spaces/triangle_params.h
#        src/scene/parameters/spaces/object_type.h
#        src/scene/parameters/spaces/object_params_pointer.h
#
#        src/image.h src/image.cu
#        src/cuda_error.h src/cuda_error.cu
#        src/render.h src/render.cu
#        src/create_scene.h src/create_scene.cu
#        src/destroy_scene.h src/destroy_scene.cu
#
#        src/test.cu
#        )

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET vecmath PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(main PUBLIC -G -g)
target_compile_options(main PUBLIC --std=c++17)
#target_compile_options(test PUBLIC --std=c++17)


target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
#target_link_libraries(test ${CUDA_LIBRARIES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC src)

#TARGET_LINK_LIBRARIES(test vecmath)
#TARGET_INCLUDE_DIRECTORIES(test PUBLIC src)
