#!/usr/bin/env python3
# 导入 rospy 库用于 ROS 节点创建与操作
import rospy
# 导入自定义消息类型，注意路径是功能包名 + 消息名
from package2.msg import TurtleSpeed  

def speed_publisher():
    # 初始化节点，命名为 turtle_speed_publisher
    rospy.init_node('turtle_speed_publisher', anonymous=True)  
    # 创建 Publisher，发布话题为 /turtle_speed，消息类型是 TurtleSpeed，队列长度 10
    pub = rospy.Publisher('/turtle_speed', TurtleSpeed, queue_size=10)  
    # 设置发布频率为 1Hz
    rate = rospy.Rate(1)  
    while not rospy.is_shutdown():
        # 实例化自定义消息并赋值
        speed_msg = TurtleSpeed()
        speed_msg.linear_x = 1.0  # 设定线速度 x 方向为 1.0
        speed_msg.angular_z = 0.5  # 先设角速度，基础作业 2 用（也可后续在 yaml 改）
        # 发布消息
        pub.publish(speed_msg)  
        # 打印发布的消息内容，方便可视化查看
        rospy.loginfo("Published turtle speed: linear_x=%f, angular_z=%f", 
                      speed_msg.linear_x, speed_msg.angular_z)
        # 按照设定频率休眠
        rate.sleep()  

if __name__ == '__main__':
    try:
        speed_publisher()
    except rospy.ROSInterruptException:
        pass