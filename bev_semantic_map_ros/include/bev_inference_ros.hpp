
#pragma once

// C++
#include <iostream>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_msgs/GridMapInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_core/GridMap.hpp>
#include <image_transport/image_transport.h>
#include <compressed_image_transport/compressed_subscriber.h>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

// PCL
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// pybind
#include <pybind11_catkin/pybind11/eigen.h>
#include <pybind11_catkin/pybind11/embed.h>
#include <pybind11_catkin/pybind11/numpy.h>
#include <pybind11_catkin/pybind11/stl.h>

#include "utilities.hpp"
#include <filters/filter_chain.hpp>

namespace py = pybind11;

using Matrix4d = Eigen::Matrix4d;
using Matrix3d = Eigen::Matrix3d;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

typedef grid_map::GridMap GridMap;
typedef grid_map_msgs::GridMap GridMapMsg;
typedef std::vector<sensor_msgs::PointCloud2ConstIterator<float>> pclConstItrVec;

class BevInferenceROS
{
public:
  BevInferenceROS(ros::NodeHandle& node, ros::NodeHandle& private_node)
    : nh_(node)
    , pnh_(private_node)
    , it_(private_node)
    , filterChain_("grid_map::GridMap"){};
  ~BevInferenceROS(){};  // Destructor

  // Functions
  bool init();

private:
  // Functions

  // Read ROS Parameters
  bool readParameters();

  // Convert from sensor_msgs::pointcloud2 ptr to Eigen Matrix
  void pointCloudToMatrix(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr, bool use_semantics, RowMatrixXf& xyz,
                          RowMatrixXf& probs);
  // Create ROS pointcloud field iterators for const input type
  pclConstItrVec createPointCloud2ConstIterators(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr,
                                                 const std::vector<std::string>& pcl_labels);

  void convertToMultichannel(cv::Mat& image, std::vector<ColMatrixXf>& multichannel_img);

  // Callback Functions for the camera images
  void camFrontCb(const sensor_msgs::ImageConstPtr& cam_ptr);
  void camBackCb(const sensor_msgs::ImageConstPtr& cam_ptr);
  // void camLeftCb(const sensor_msgs::ImageConstPtr& cam_ptr);
  // void camRightCb(const sensor_msgs::ImageConstPtr& cam_ptr);

  // Callback Functions for the camera info topics
  // void infoLeftCb(const sensor_msgs::CameraInfoConstPtr& info_ptr);
  // void infoRightCb(const sensor_msgs::CameraInfoConstPtr& info_ptr);
  void infoFrontCb(const sensor_msgs::CameraInfoConstPtr& info_ptr);
  void infoBackCb(const sensor_msgs::CameraInfoConstPtr& info_ptr);

  // // Velodyne Cloud callback
  // void velodyneCb(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr);

  // Merged Cloud callback
  void mergedCb(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr);

  // Raw Elevation Map callback
  void elevationMapCb(const grid_map_msgs::GridMapConstPtr& msg);

  // // Voxel Map callback (For GVOM Cloud) (TODO)
  // void gvomCb(const sensor_msgs::PointCloud2ConstPtr& gvom_ptr);

  // Timer-based callback for inference
  void inferenceCb();

  // Timer-based callback for camera info initialization
  void camInfoInitCb();

  // Publish Grid Map
  void publishMaps(GridMap& map_1, GridMap& map_2, GridMap& map_3);

  // Variables
  ros::NodeHandle nh_, pnh_;           // ROS Nodehandles
  ros::Subscriber subCloud_;  // ROS Subscriber for merged velodyne Pointcloud
  image_transport::Subscriber subCamFront_, subCamBack_;  // ROS Subscribers for the images
  ros::Subscriber subCamInfoFront_, subCamInfoBack_;                          // ROS Subscribers for the Camera Infos
  ros::Publisher pubGridMap_, pubGridMapInfo_;  // ROS Publisher for output of network
  ros::Timer inferenceTimer_;                   // Timer for Network inference
  ros::Timer camInfoTimer_;                     // Timer for Network inference
  image_transport::ImageTransport it_;          // Image transport
  ros::Subscriber subElevationMap_;             // ROS Subscriber for Raw elevation Map

  // ros::Subscriber subGvom_;  // ROS Subscriber for GVOM cloud

  // Transform lookup
  // Does it make sense to have multiple buffers for each thread ?

  // Buffer for Cameras
  std::unique_ptr<tf2_ros::Buffer> tfBufferFC_, tfBufferBC_;
  std::unique_ptr<tf2_ros::TransformListener> tfListenerFC_, tfListenerBC_;

  // Buffer for Cloud
  std::unique_ptr<tf2_ros::Buffer> tfBufferC_;
  std::unique_ptr<tf2_ros::TransformListener> tfListenerC_;

  // Mutex
  std::mutex cloudMtx_, elevationMapMtx_, frontMtx_, backMtx_;

  // Booleans for message
  bool camInfoFrontRec_ = false;
  bool camInfoBackRec_ = false;
  bool camInfoInit_ = false;
  bool camFrontRec_ = false;
  bool camBackRec_ = false;
  bool cloudRec_ = false;
  bool gridmapRec_ = false;

  // Eigen Matrices for the Projection Matrices (P)
  // Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_left_;
  // Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_right_;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_front_;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_back_;

  // // Image Matrices
  // cv::Mat frontImage_;
  // cv::Mat leftImage_;
  // cv::Mat rightImage_;
  // cv::Mat backImage_;

  // Image Vectors
  std::vector<ColMatrixXf> frontImage_;
  // std::vector<ColMatrixXf> leftImage_;
  // std::vector<ColMatrixXf> rightImage_;
  std::vector<ColMatrixXf> backImage_;

  // Transforms
  Matrix4d T_sensor_gravity__map_ = Matrix4d::Identity();
  Matrix4d T_map__base_link_ = Matrix4d::Identity();
  Matrix4d T_map__grid_map_center_ = Matrix4d::Identity();
  Matrix4d T_map__front_cam_link_ = Matrix4d::Identity();
  // Matrix4d T_map__left_cam_link_ = Matrix4d::Identity();
  // Matrix4d T_map__right_cam_link_ = Matrix4d::Identity();
  Matrix4d T_map__back_cam_link_ = Matrix4d::Identity();
  Matrix4d T_sensor_origin_link__map_ = Matrix4d::Identity();

  // PCD Data
  std::vector<std::string> pclFieldNames_{ "x", "y", "z" };  // Vector of geometric field names
  RowMatrixXf pcd_ = RowMatrixXf::Zero(1000, 3);
  // PCL semantic labels
  std::vector<std::string> probFieldNames_{ "grass", "tree" };  // Vector of semantic probability field names
  std::map<std::string, int> probFieldMap_;
  bool useProbOther_ = true;

  // GVOM Data
  // RowMatrixXf gvom_ = RowMatrixXf::Zero(10000, 3);

  // Grid Map Data
  GridMapMsg::ConstPtr ele_msg_;
  GridMap map_memory_;

  // Vectors for passing data to python

  // Image Data
  std::vector<std::vector<ColMatrixXf>> imageVect_;
  std::vector<Matrix3d> kVect_;
  std::vector<Matrix3d> imageRotVect_;
  std::vector<Eigen::Matrix<double, 3, 1>> imageTransVect_;
  uint64_t imgTs_ = 0;

  // Params
  std::string mapFrame_ = "map_o3d_localization_manager";
  std::string egoFrame_ = "base_inverted";
  int infFreq_ = 10;
  int pclMinPts_ = 1000;
  int voxelMapMinPts_ = 1000;
  int resizeWidth_ = 640;
  int resizeHeight_ = 396;

  int mapWidth_ = 500;  // gridmap width and height
  float res_ = 0.2;     // Resolution of micro range map
  // float lengthMicro_ = 100; // Resolution * Width

  // int mapWidthShort_ = 400;  // gridmap width and height
  // float resShort_ = 0.5;     // Resolution of short range map
  // float lengthShort_ =

  // Python handler for pybind
  py::object pyHandle_;

  // Filter chain object
  filters::FilterChain<grid_map::GridMap> filterChain_;
  // filters::FilterChain<grid_map::GridMap> filterChainMicro_;

  //! Filter chain parameters name.
  std::string filterChainParametersName_;
  // std::string filterChainParametersNameShort_;
};