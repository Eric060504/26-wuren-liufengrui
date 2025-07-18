cmake_minimum_required(VERSION 3.10)
project(Test)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(CMake\ 1/common)
add_subdirectory(CMake\ 1/modules)

add_executable(test CMake\ 1/main.cpp)

target_link_libraries(test 
    M1 
    M2 
    A1 
    A2 
    math 
    kalman 
    ${OpenCV_LIBS}
)

target_include_directories(test PUBLIC
    CMake\ 1/modules/M1/include
    CMake\ 1/modules/M2/include
    CMake\ 1/modules/A1/include
    CMake\ 1/modules/A2/include
    CMake\ 1/common/math/include
    CMake\ 1/common/kalman/include
)