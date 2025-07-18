# common/math/CMakeLists.txt
aux_source_directory(src Path_Math)

# 生成名为math的静态库（最终会生成libmath.a）
add_library(math STATIC ${Path_Math})

target_include_directories(math PUBLIC include)
target_link_libraries(math PUBLIC ${OpenCV_LIBS})