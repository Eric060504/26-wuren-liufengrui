cmake_minimum_required(VERSION 2.8.12)
project(fsd_common_msgs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  message_generation  # Needed to generate custom message types
  std_msgs
  roscpp
  rospy
  geometry_msgs
  can_msgs
  nav_msgs
  )

add_message_files(  # Tell catkin which custom message files to include
  FILES
  Cone.msg
  ConeDetections.msg
  Map.msg
  ControlCommand.msg
  CarState.msg
  CarStateDt.msg
  Mission.msg
  TrajectoryPoint.msg
  EchievMessage.msg
  RemoteControlCommand.msg
  AsensingMessage.msg
  AsState.msg
  ResAndAmi.msg
  Time.msg
  Feedback.msg
  Visualization.msg
  DecisionFlag.msg
  DrivingDynamics.msg
  CanFrames.msg
  ConeDbscan.msg
  ConeDetectionsDbscan.msg
  YoloCone.msg
  YoloConeDetections.msg
  YoloConeTrack.msg
  YoloConeDetectionsTrack.msg
  InsDelta.msg
  SkidpadGlobalCenterLine.msg
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  can_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs can_msgs# Needed to generate custom messages
)
include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
  ${rospy_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(cone_counter src/count.cpp)
add_executable(cone_visualizer src/visualize.cpp)

add_dependencies(cone_counter 
  ${${PROJECT_NAME}_EXPORTED_TARGETS}  # 自定义消息生成目标
  ${catkin_EXPORTED_TARGETS}          # 依赖包的消息/服务生成目标
)
add_dependencies(cone_visualizer 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

# 链接库
target_link_libraries(cone_counter ${catkin_LIBRARIES})
target_link_libraries(cone_visualizer ${catkin_LIBRARIES})

# 安装可执行文件
install(TARGETS
    cone_counter
    cone_visualizer
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

