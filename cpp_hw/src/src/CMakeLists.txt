cmake_minimum_required(VERSION 3.0.2)
project(merged_package)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  fsd_common_msgs
  Eigen
  pcl_conversions
)

find_package(PCL 1.10 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)

add_executable(point_cloud_processor src/carla.cpp)
add_executable(cone_node src/cone.cpp)

target_link_libraries(point_cloud_processor
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(cone_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

set_property(TARGET cone_node PROPERTY CXX_STANDARD 11)
set_property(TARGET cone_node PROPERTY CXX_STANDARD_REQUIRED ON)