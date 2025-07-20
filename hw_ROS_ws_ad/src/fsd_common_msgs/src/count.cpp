#include <ros/ros.h>
#include <fsd_common_msgs/ConeDetections.h>

void coneDetectionsCallback(const fsd_common_msgs::ConeDetections::ConstPtr& msg) {
    int blue_count = 0;
    int red_count = 0;

    for (const auto& cone : msg->cone_detections) {
        if (cone.color.data == "b") {
            blue_count++;
        } else if (cone.color.data == "r") {
            red_count++;
        }
    }

    ROS_INFO("Blue cones: %d", blue_count);
    ROS_INFO("Red cones: %d", red_count);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cone_counter");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<fsd_common_msgs::ConeDetections>(
        "/perception/lidar/cone_side", 10, coneDetectionsCallback);

    ros::spin();

    return 0;
}