#include <ros/ros.h>
#include <fsd_common_msgs/ConeDetections.h>
#include <visualization_msgs/Marker.h>

ros::Publisher marker_pub;

void coneDetectionsCallback(const fsd_common_msgs::ConeDetections::ConstPtr& msg) {
    visualization_msgs::Marker cone_marker;
    cone_marker.header.frame_id = "map";
    cone_marker.header.stamp = ros::Time::now();
    cone_marker.ns = "cone_visualization";
    cone_marker.type = visualization_msgs::Marker::CYLINDER;
    cone_marker.action = visualization_msgs::Marker::ADD;
    cone_marker.scale.x = 0.2;
    cone_marker.scale.y = 0.2;
    cone_marker.scale.z = 0.3;
    cone_marker.pose.orientation.w = 1.0;
    cone_marker.lifetime = ros::Duration();

    int id = 0;

    for (const auto& cone : msg->cone_detections) {
        cone_marker.id = id++;
        cone_marker.pose.position.x = cone.position.x;
        cone_marker.pose.position.y = cone.position.y;
        cone_marker.pose.position.z = 0.15;

        if (cone.color.data == "b") {
            cone_marker.color.r = 0.0;
            cone_marker.color.g = 0.0;
            cone_marker.color.b = 1.0;
        } else if (cone.color.data == "r") {
            cone_marker.color.r = 1.0;
            cone_marker.color.g = 0.0;
            cone_marker.color.b = 0.0;
        } else {
            continue;  // 忽略其他颜色
        }

        cone_marker.color.a = 1.0;

        marker_pub.publish(cone_marker);
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cone_visualizer");
    ros::NodeHandle nh;

    marker_pub = nh.advertise<visualization_msgs::Marker>("cone_marker", 10);
    ros::Subscriber sub = nh.subscribe<fsd_common_msgs::ConeDetections>(
        "/perception/lidar/cone_side", 10, coneDetectionsCallback);

    ros::spin();

    return 0;
}