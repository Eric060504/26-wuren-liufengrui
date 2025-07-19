#include "ros/ros.h"
// 导入自定义消息头文件，由之前的 msg 生成，路径规则：功能包名/消息名.h
#include "package2/TurtleSpeed.h"  
// 用于控制乌龟运动的话题消息类型（ turtlesim 功能包的）
#include "geometry_msgs/Twist.h"  

// 定义发布乌龟控制指令的 Publisher
ros::Publisher turtle_cmd_pub;  

// 回调函数，处理接收到的自定义速度消息
void speedCallback(const package2::TurtleSpeed::ConstPtr& msg)
{
    double angular_z;
    // 从参数服务器获取角速度参数，参数名要和 yaml 里一致，若没找到用默认值 0.5
   if(!ros::param::get("~angular_z", angular_z)){
        angular_z = 0.5;  
    }  

    geometry_msgs::Twist cmd;
    // 线速度从自定义消息获取
    cmd.linear.x = msg->linear_x;  
    // 角速度从参数服务器获取
    cmd.angular.z = angular_z;  

    // 发布控制指令，让乌龟运动
    turtle_cmd_pub.publish(cmd);  
    ROS_INFO("Received speed: linear_x=%f, Using angular_z=%f to control turtle", 
             msg->linear_x, angular_z);
}

int main(int argc, char **argv)
{
    // 初始化节点
    ros::init(argc, argv, "turtle_controller");  
    ros::NodeHandle n;
 

    // 创建 Subscriber，订阅 /turtle_speed 话题，消息类型是自定义的 TurtleSpeed，
    // 队列长度 10，绑定回调函数 speedCallback
    ros::Subscriber sub = n.subscribe("/turtle_speed", 10, speedCallback);  

    // 创建 Publisher，发布 /turtle1/cmd_vel 话题（turtlesim 控制乌龟运动的话题），
    // 消息类型是 geometry_msgs::Twist，队列长度 10
    turtle_cmd_pub = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel", 10);  

    // 循环处理回调
    ros::spin();  

    return 0;
}