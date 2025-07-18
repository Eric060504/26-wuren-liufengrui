aux_source_directory(src Path_A1)

add_library(A1 STATIC ${Path_A1})

target_include_directories(A1 PUBLIC include)