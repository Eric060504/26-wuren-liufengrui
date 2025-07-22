#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/io/pcd_io.h>
#include <memory> 

class PointCloudProcessor
{
public:
    PointCloudProcessor(ros::NodeHandle& nh) : nh_(nh)
    {
        // 创建发布者
        pub_ = nh_.advertise<sensor_msgs/PointCloud2>("/lidar_points", 1);
    }

    void processBagFile(const std::string& bag_file_path)
    {
        // 打开ROS bag文件
        rosbag::Bag bag;
        try
        {
            bag.open(bag_file_path, rosbag::bagmode::Read);
        }
        catch (rosbag::BagException e)
        {
            ROS_ERROR("Failed to open bag file: %s", bag_file_path.c_str());
            return;
        }

        // 定义需要读取的话题
        std::vector<std::string> topics;
        topics.push_back("/carla/ego_vehicle/lidar_mid");
        topics.push_back("/carla/ego_vehicle/lidar_up");
        topics.push_back("/carla/ego_vehicle/lidar_down");

        // 读取bag文件中的点云数据
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // 使用智能指针管理点云
        auto cloud_all = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        foreach (rosbag::MessageInstance const m, view)
        {
            sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (cloud_msg != nullptr)
            {
                pcl::PointCloud<pcl::PointXYZI> cloud;
                pcl::fromROSMsg(*cloud_msg, cloud);

                // 将当前点云添加到总点云中
                *cloud_all += cloud;
            }
        }

        bag.close();

        // 修改强度值
        for (auto& point : *cloud_all)
        {
            // 将强度从0~1映射到1~256
            point.intensity = std::min(255.0f, std::max(1.0f, point.intensity * 255.0f));
        }

        // 将处理后的点云发布出去
        sensor_msgs::PointCloud2 output_cloud;
        pcl::toROSMsg(*cloud_all, output_cloud);
        output_cloud.header.frame_id = "lidar";
        pub_.publish(output_cloud);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_processor");
    ros::NodeHandle nh;

    PointCloudProcessor processor(nh);

    // 替换为你的bag文件路径
    std::string bag_file_path = "~/cpp_hw/bag/carla_test.bag";

    processor.processBagFile(bag_file_path);

    ros::spin();
    return 0;
}