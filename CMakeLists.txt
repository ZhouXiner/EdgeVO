cmake_minimum_required(VERSION 3.2)
project(EdgeVO)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE release)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include_directories("/usr/include/eigen3")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
#yaml cpp
FIND_PACKAGE(yaml-cpp REQUIRED)
INCLUDE_DIRECTORIES(${YAMLCPP_INCLUDE_DIR})

set(THIRD_LIBS
        #${PROJECT_SOURCE_DIR}/3rdparty/lib/libyaml-cpp.so.0.6.3
        ${YAMLCPP_LIBRARY}
        #yaml-cpp
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${CERES_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY})

set(UTILITY_HEADER_FILES "Utils/LGSX.h" "Utils/Utility.h")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(EdgeVO main.cpp ${UTILITY_HEADER_FILES})
target_link_libraries(EdgeVO EdgeVO_LIBS ${THIRD_LIBS})