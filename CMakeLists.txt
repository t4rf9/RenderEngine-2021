CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(main LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75-real)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

ADD_SUBDIRECTORY(deps/vecmath)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

ADD_EXECUTABLE(main
        src/camera/camera.h src/camera/camera.cu
        src/objects/group.h src/objects/group.cu
        src/lights/lights.h
        src/lights/light.h src/lights/light.cu
        src/lights/directional_light.h src/lights/directional_light.cu
        src/lights/point_light.h src/lights/point_light.cu
        src/material.h
        src/objects/mesh.h src/objects/mesh.cu
        src/objects/object3d.h src/objects/object3d.cu
        src/objects/plane.h src/objects/plane.cu
        src/objects/revsurface.h src/objects/revsurface.cu
        src/objects/curve.h src/objects/curve.cu
        src/objects/BezierCurve.h src/objects/BezierCurve.cu
        src/objects/BsplineCurve.h src/objects/BsplineCurve.cu
        src/objects/sphere.h src/objects/sphere.cu
        src/objects/transform.h src/objects/transform.cu
        src/objects/triangle.h src/objects/triangle.cu
        src/objects/BoundingBox.h src/objects/BoundingBox.cu
        src/objects/BoundingObject.h src/objects/BoundingObject.cu
        src/hit.h src/hit.cu
        src/image.h src/image.cu
        src/ray.h src/ray.cu
        src/scene_parser.h src/scene_parser.cu
        src/scene.h src/scene.cu
        src/cuda_error.h src/cuda_error.cu
        src/main.cu
        src/camera/perspective_camera.cu src/camera/perspective_camera.h
        src/camera/cameras.h src/material.cu
        )

ADD_EXECUTABLE(test
        src/camera/camera.h src/camera/camera.cu
        src/objects/group.h src/objects/group.cu
        src/lights/lights.h
        src/lights/light.h src/lights/light.cu
        src/lights/directional_light.h src/lights/directional_light.cu
        src/lights/point_light.h src/lights/point_light.cu
        src/material.h
        src/objects/mesh.h src/objects/mesh.cu
        src/objects/object3d.h src/objects/object3d.cu
        src/objects/plane.h src/objects/plane.cu
        src/objects/revsurface.h src/objects/revsurface.cu
        src/objects/curve.h src/objects/curve.cu
        src/objects/BezierCurve.h src/objects/BezierCurve.cu
        src/objects/BsplineCurve.h src/objects/BsplineCurve.cu
        src/objects/sphere.h src/objects/sphere.cu
        src/objects/transform.h src/objects/transform.cu
        src/objects/triangle.h src/objects/triangle.cu
        src/objects/BoundingBox.h src/objects/BoundingBox.cu
        src/objects/BoundingObject.h src/objects/BoundingObject.cu
        src/hit.h src/hit.cu
        src/image.h src/image.cu
        src/ray.h src/ray.cu
        src/scene_parser.h src/scene_parser.cu
        src/scene.h src/scene.cu
        src/cuda_error.h src/cuda_error.cu
        src/test.cu
        src/camera/perspective_camera.cu src/camera/perspective_camera.h
        src/camera/cameras.h src/material.cu
        )

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET vecmath PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(test ${CUDA_LIBRARIES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC src)

TARGET_LINK_LIBRARIES(test vecmath)
TARGET_INCLUDE_DIRECTORIES(test PUBLIC src)
