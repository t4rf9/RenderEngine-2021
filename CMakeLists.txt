CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(main LANGUAGES CXX)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

ADD_SUBDIRECTORY(deps/vecmath)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

ADD_EXECUTABLE(main
        src/camera/camera.h src/camera/camera.cpp
        src/objects/group.h src/objects/group.cpp
        src/lights/lights.h
        src/lights/light.h src/lights/light.cpp
        src/lights/directional_light.h src/lights/directional_light.cpp
        src/lights/point_light.h src/lights/point_light.cpp
        src/material.h
        src/objects/mesh.h src/objects/mesh.cpp
        src/objects/object3d.h src/objects/object3d.cpp
        src/objects/plane.h src/objects/plane.cpp
        src/objects/revsurface.h src/objects/revsurface.cpp
        src/objects/curve.h src/objects/curve.cpp
        src/objects/BezierCurve.h src/objects/BezierCurve.cpp
        src/objects/BsplineCurve.h src/objects/BsplineCurve.cpp
        src/objects/sphere.h src/objects/sphere.cpp
        src/objects/transform.h src/objects/transform.cpp
        src/objects/triangle.h src/objects/triangle.cpp
        src/objects/BoundingBox.h src/objects/BoundingBox.cpp
        src/objects/BoundingObject.h src/objects/BoundingObject.cpp
        src/hit.h src/hit.cpp
        src/image.h src/image.cpp
        src/ray.h src/ray.cpp
        src/scene_parser.h src/scene_parser.cpp
        src/scene.h src/scene.cpp
        src/main.cpp
        src/camera/perspective_camera.cpp src/camera/perspective_camera.h
        src/camera/cameras.h src/material.cpp
        )

ADD_EXECUTABLE(test
        src/camera/camera.h src/camera/camera.cpp
        src/objects/group.h src/objects/group.cpp
        src/lights/lights.h
        src/lights/light.h src/lights/light.cpp
        src/lights/directional_light.h src/lights/directional_light.cpp
        src/lights/point_light.h src/lights/point_light.cpp
        src/material.h
        src/objects/mesh.h src/objects/mesh.cpp
        src/objects/object3d.h src/objects/object3d.cpp
        src/objects/plane.h src/objects/plane.cpp
        src/objects/revsurface.h src/objects/revsurface.cpp
        src/objects/curve.h src/objects/curve.cpp
        src/objects/BezierCurve.h src/objects/BezierCurve.cpp
        src/objects/BsplineCurve.h src/objects/BsplineCurve.cpp
        src/objects/sphere.h src/objects/sphere.cpp
        src/objects/transform.h src/objects/transform.cpp
        src/objects/triangle.h src/objects/triangle.cpp
        src/objects/BoundingBox.h src/objects/BoundingBox.cpp
        src/objects/BoundingObject.h src/objects/BoundingObject.cpp
        src/hit.h src/hit.cpp
        src/image.h src/image.cpp
        src/ray.h src/ray.cpp
        src/scene_parser.h src/scene_parser.cpp
        src/scene.h src/scene.cpp
        src/test.cpp
        src/camera/perspective_camera.cpp src/camera/perspective_camera.h
        src/camera/cameras.h src/material.cpp
        )

TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE src)


TARGET_LINK_LIBRARIES(test vecmath)
TARGET_INCLUDE_DIRECTORIES(test PRIVATE src)
