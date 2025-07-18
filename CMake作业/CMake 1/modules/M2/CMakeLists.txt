aux_source_directory(src Path_M2)

add_library(M2 STATIC ${Path_M2})

target_include_directories(M2 PUBLIC 
    include
    /home/eric/cmake_ws/CMake_Final/CMake作业/CMake\ 1/common/kalman/include
    )

target_link_libraries(M2 PUBLIC
    A1
    A2
    kalman
    ${OpenCV_LIBS}
)