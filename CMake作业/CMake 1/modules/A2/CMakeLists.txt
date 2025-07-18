aux_source_directory(src Path_A2)

add_library(A2 STATIC ${Path_A2})

target_include_directories(A2 PUBLIC include)