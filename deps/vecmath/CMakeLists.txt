CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(vecmath LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75-real)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(VECMATH_INCLUDES
        include/Matrix2f.h
        include/Matrix3f.h
        include/Matrix4f.h
        include/Quat4f.h
        include/vecmath.h
        include/Vector2f.h
        include/Vector3f.h
        include/Vector4f.h)

SET(VECMATH_SOURCES
        src/Matrix2f.cu
        src/Matrix3f.cu
        src/Matrix4f.cu
        src/Quat4f.cu
        src/Vector2f.cu
        src/Vector3f.cu
        src/Vector4f.cu
        )

SET(CMAKE_CXX_STANDARD 17)

ADD_LIBRARY(${PROJECT_NAME} STATIC ${VECMATH_INCLUDES} ${VECMATH_SOURCES})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC include)
