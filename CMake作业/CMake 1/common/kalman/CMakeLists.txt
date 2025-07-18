find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(kalman_INCLUDE_DIR)
list(APPEND kalman_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
list(APPEND kalman_INCLUDE_DIR "${OpenCV_INCLUDE_DIRS}")

set(kalman_INCLUDE_DIRS ${kalman_INCLUDE_DIR}
                        CACHE PATH "kalman filter include directories")


# 定义接口库（仅头文件），库名为kalman
add_library(kalman INTERFACE)

# 公开头文件目录
target_include_directories(kalman INTERFACE 
  ${CMAKE_CURRENT_LIST_DIR}/include 
  ${OpenCV_INCLUDE_DIRS}
)

# 链接依赖
target_link_libraries(kalman INTERFACE ${OpenCV_LIBS})