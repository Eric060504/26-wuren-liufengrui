#查找并设置 catkin 以及所需的依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  turtlesim
  message_generation  # 新增 message_generation 依赖
)

project(package2)

add_message_files(
  FILES
  TurtleSpeed.msg  # 新增 TurtleSpeed 消息文件
)

generate_messages(
  DEPENDENCIES
  std_msgs  # 确保 std_msgs 作为依赖
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs turtlesim message_runtime  # 新增 message_runtime 依赖
)

include_directories(
  ${catkin_INCLUDE_DIRS}
) # 包含 catkin 的头文件目录

add_executable(turtle_controller src/turtle_controller.cpp) # 新增可执行文件
target_link_libraries(turtle_controller ${catkin_LIBRARIES}) # 链接 catkin 库
add_dependencies(turtle_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) # 确保依赖关系正确