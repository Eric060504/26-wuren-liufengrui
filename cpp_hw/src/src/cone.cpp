#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>

// 内参矩阵
Eigen::Matrix3d K;
// 外参矩阵
Eigen::Matrix<double, 3, 4> T;

void coneDetectionCallback(const fsd_common_msgs::ConeDetections::ConstPtr& msg) {
  // 创建CV图像
  auto image = std::make_shared<cv::Mat>(cv::Mat::zeros(360, 1280, CV_8UC3));
  
  for (const auto& detection : msg->cones) {
    // 提取锥桶3D坐标
    Eigen::Vector4d point_3d(detection.position.x, detection.position.y, detection.position.z, 1.0);
    
    // 应用外参矩阵
    Eigen::Vector3d point_cam = T * point_3d;
    
    // 应用内参矩阵
    Eigen::Vector3d point_pixel = K * point_cam;
    
    // 归一化
    point_pixel /= point_pixel(2);
    
    // 转换为像素坐标
    int u = static_cast<int>(point_pixel(0));
    int v = static_cast<int>(point_pixel(1));
    
    // 检查坐标是否在图像范围内
    if (u >= 0 && u < 1280 && v >= 0 && v < 360) {
      // 根据锥桶颜色设置绘制颜色
      cv::Scalar color;
      switch (detection.color) {
        case fsd_common_msgs::Cone::COLOR_RED:
          color = cv::Scalar(0, 0, 255);  // OpenCV使用BGR格式
          break;
        case fsd_common_msgs::Cone::COLOR_BLUE:
          color = cv::Scalar(255, 0, 0);
          break;
        case fsd_common_msgs::Cone::COLOR_YELLOW:
          color = cv::Scalar(0, 255, 255);
          break;
        default:
          color = cv::Scalar(255, 255, 255);  // 默认白色
          break;
      }
      
      // 绘制实心圆表示锥桶投影点
      cv::circle(*image, cv::Point(u, v), 5, color, -1);
    }
  }
  
  // 将CV图像转换为ROS图像消息
  auto img_msg = std::make_shared<sensor_msgs::Image>(cv_bridge::CvImage(std_msgs::Header(), "bgr8", *image).toImageMsg());
  img_msg->header = msg->header;  // 使用相同的header以便在Rviz中对齐
  
  // 发布图像
  static image_transport::Publisher image_pub = *std::static_pointer_cast<image_transport::Publisher>(
      ros::NodeHandle().getNodeHandlePtr()->getPublisher("/projected_image"));
  image_pub.publish(img_msg);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "cone_projection_node");
  
  auto nh = std::make_shared<ros::NodeHandle>();
  
  // 初始化内参矩阵
  K << 532.795, 0.0, 632.15,
       0.0, 532.72, -3.428,
       0.0, 0.0, 1.0;

  // 初始化外参矩阵
  T << 3.5594209875121074e-03, -9.9987761481865733e-01, -1.5234365979146680e-02, 8.9277270417879417e-02,
       1.9781062410225703e-03, 1.5241472820252011e-02, -9.9988188532544631e-01, 9.1100499695349946e-01,
       9.9999170877459420e-01, 3.5288653732390984e-03, 2.0321149683686368e-03, 1.9154049062915668e+00;
  
  // 创建图像发布者
  auto it = std::make_shared<image_transport::ImageTransport>(*nh);
  static image_transport::Publisher image_pub = it->advertise("/projected_image", 1);
  
  // 订阅锥桶检测话题
  ros::Subscriber sub = nh->subscribe("/perception/lidar/cone_detections", 1, coneDetectionCallback);
  
  ros::spin();
  
  return 0;
}