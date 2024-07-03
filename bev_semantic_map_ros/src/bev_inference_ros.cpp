#include "bev_inference_ros.hpp"
#include <chrono>
#include <thread>

bool BevInferenceROS::init()
{
  // Transform Listener
  tfBufferFC_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerFC_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferFC_);

  tfBufferLC_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerLC_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferLC_);

  tfBufferRC_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerRC_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferRC_);

  tfBufferBC_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerBC_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferBC_);

  tfBufferV_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerV_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferV_);

  tfBufferG_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10.0));
  tfListenerG_ = std::make_unique<tf2_ros::TransformListener>(*tfBufferG_);

  if (readParameters() == false)
  {
    ROS_ERROR("Failed to read parameters");
    return false;
  }

  // Setup filter chain.
  if (!filterChainShort_.configure(filterChainParametersNameShort_, pnh_))
  {
    ROS_ERROR("Could not configure the filter chain!");
    return false;
  }
  if (!filterChainMicro_.configure(filterChainParametersNameMicro_, pnh_))
  {
    ROS_ERROR("Could not configure the filter chain!");
    return false;
  }

  {
    py::gil_scoped_acquire acquire;

    // Import all python modules and initialize the model with pybind
    // Import python module
    auto sys = py::module::import("sys");
    auto path = sys.attr("path");
    std::string module_path = ros::package::getPath("perception_bev_learning_ros");
    module_path = module_path + "/scripts";
    std::cout << "Initializing Python" << std::endl;

    path.attr("insert")(0, module_path);
    py::module bev_module = py::module::import("bev_inference");

    // Import python class
    pyHandle_ = bev_module.attr("BevInference");

    std::cout << "XXXXXXXXXXXX" << std::endl;
    // Initialize parameters
    pyHandle_.attr("__init__")(pyHandle_);

    py::gil_scoped_release release;
  }

  // Publishers
  pubGridMapMicro_ = pnh_.advertise<GridMapMsg>("bev_trav_map_micro", 1);
  pubGridMapMicroInfo_ = pnh_.advertise<grid_map_msgs::GridMapInfo>("bev_map_micro_info", 1);

  pubGridMapShort_ = pnh_.advertise<GridMapMsg>("bev_trav_map_short", 1);
  pubGridMapShortInfo_ = pnh_.advertise<grid_map_msgs::GridMapInfo>("bev_map_short_info", 1);

  pubGridMapMicroVel_ = pnh_.advertise<GridMapMsg>("bev_trav_map_micro_vel", 1);
  pubGridMapMicroVelInfo_ = pnh_.advertise<grid_map_msgs::GridMapInfo>("bev_map_micro_vel_info", 1);

  // Camera Info Subscribers
  subCamInfoLeft_ = pnh_.subscribe<sensor_msgs::CameraInfo>("left/camera_info", 2, &BevInferenceROS::infoLeftCb, this,
                                                            ros::TransportHints().tcpNoDelay());
  subCamInfoRight_ = pnh_.subscribe<sensor_msgs::CameraInfo>("right/camera_info", 2, &BevInferenceROS::infoRightCb,
                                                             this, ros::TransportHints().tcpNoDelay());
  subCamInfoFront_ = pnh_.subscribe<sensor_msgs::CameraInfo>("front/camera_info", 2, &BevInferenceROS::infoFrontCb,
                                                             this, ros::TransportHints().tcpNoDelay());
  subCamInfoBack_ = pnh_.subscribe<sensor_msgs::CameraInfo>("back/camera_info", 2, &BevInferenceROS::infoBackCb, this,
                                                            ros::TransportHints().tcpNoDelay());

  // Camera Image Subscribers (Rectified Images)
  subCamFront_ = it_.subscribe("front/rectified_image", 2, &BevInferenceROS::camFrontCb, this);
  subCamBack_ = it_.subscribe("back/rectified_image", 2, &BevInferenceROS::camBackCb, this);
  subCamLeft_ = it_.subscribe("left/rectified_image", 2, &BevInferenceROS::camLeftCb, this);
  subCamRight_ = it_.subscribe("right/rectified_image", 2, &BevInferenceROS::camRightCb, this);

  // Velodyne Cloud Subscriber
  subVelodyneMerged_ = pnh_.subscribe<sensor_msgs::PointCloud2>("cloud", 2, &BevInferenceROS::velodyneCb, this,
                                                                ros::TransportHints().tcpNoDelay());

  // VoxelMap Subscriber
  subGvom_ = pnh_.subscribe<sensor_msgs::PointCloud2>("gvomcloud", 2, &BevInferenceROS::gvomCb, this,
                                                      ros::TransportHints().tcpNoDelay());

  // Raw ElevationMap Subscriber (Gridmap)
  subElevationMap_ = pnh_.subscribe<grid_map_msgs::GridMap>("raw_elevation_map", 1, &BevInferenceROS::elevationMapCb,
                                                            this, ros::TransportHints().tcpNoDelay());

  // Timer Callback
  inferenceTimer_ = pnh_.createTimer(ros::Duration(1.0 / infFreq_), std::bind(&BevInferenceROS::inferenceCb, this));
  // Timer Callback
  camInfoTimer_ = pnh_.createTimer(ros::Duration(0.2), std::bind(&BevInferenceROS::camInfoInitCb, this));

  ROS_INFO("Initialized");
  return true;
}

bool BevInferenceROS::readParameters()
{
  pnh_.param("filter_chain_short", filterChainParametersNameShort_, std::string("grid_map_filters_short"));
  pnh_.param("filter_chain_micro", filterChainParametersNameMicro_, std::string("grid_map_filters_micro"));
  return true;
}

void BevInferenceROS::infoLeftCb(const sensor_msgs::CameraInfoConstPtr& info_ptr)
{
  // Extract P matrix elements from the CameraInfo message
  const std::vector<double> P_data(info_ptr->P.begin(), info_ptr->P.end());
  // Populate the class variable P from the K matrix data
  P_left_ = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(P_data.data());
  subCamInfoLeft_.shutdown();
  camInfoLeftRec_ = true;
}

void BevInferenceROS::infoRightCb(const sensor_msgs::CameraInfoConstPtr& info_ptr)
{
  // Extract P matrix elements from the CameraInfo message
  const std::vector<double> P_data(info_ptr->P.begin(), info_ptr->P.end());
  // Populate the class variable P from the K matrix data
  P_right_ = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(P_data.data());
  subCamInfoRight_.shutdown();
  camInfoRightRec_ = true;
}

void BevInferenceROS::infoFrontCb(const sensor_msgs::CameraInfoConstPtr& info_ptr)
{
  // Extract P matrix elements from the CameraInfo message
  const std::vector<double> P_data(info_ptr->P.begin(), info_ptr->P.end());
  // Populate the class variable P from the K matrix data
  P_front_ = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(P_data.data());
  subCamInfoFront_.shutdown();
  camInfoFrontRec_ = true;
}

void BevInferenceROS::infoBackCb(const sensor_msgs::CameraInfoConstPtr& info_ptr)
{
  // Extract P matrix elements from the CameraInfo message
  const std::vector<double> P_data(info_ptr->P.begin(), info_ptr->P.end());
  // Populate the class variable P from the K matrix data
  P_back_ = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(P_data.data());
  subCamInfoBack_.shutdown();
  camInfoBackRec_ = true;
}

void BevInferenceROS::camInfoInitCb()
{
  if (camInfoLeftRec_ && camInfoRightRec_ && camInfoFrontRec_ && camInfoBackRec_)
  {
    std::cout << "All K Matrices initialized" << std::endl;
    camInfoInit_ = true;
    camInfoTimer_.stop();
  }
}

void BevInferenceROS::convertToMultichannel(cv::Mat& image, std::vector<ColMatrixXf>& multichannel_img)
{
  std::vector<cv::Mat> image_split;
  cv::split(image, image_split);
  for (auto img : image_split)
  {
    ColMatrixXf eigen_img;
    cv::cv2eigen(img, eigen_img);
    multichannel_img.push_back(eigen_img);
  }
}

void BevInferenceROS::inferenceCb()
{
  ROS_INFO("Entered Timer Callback");
  gridmapRec_ = true;
  static auto prev_ts = ros::Time(0.0);
  std::cout << camInfoInit_ << " | " << camFrontRec_ << " | " << gridmapRec_ << std::endl;
  // TODO Move the Image related callbacks, and everything in a vector so that can be handled by a for loop
  if (camInfoInit_ && camFrontRec_ && gridmapRec_ && camLeftRec_ && camRightRec_ && camBackRec_ && gvomRec_ &&
      velodyneRec_)
  {
    auto ts = ros::Time::now();
    std::cout << "ts delta:" << (ts - prev_ts).toSec() << std::endl;
    prev_ts = ts;

    auto start = std::chrono::high_resolution_clock::now();
    // Add above the conditions for first message received by all
    // Clear the Vectors
    imageVect_.clear();
    kVect_.clear();  // Can potentially not update this since it won't change
    imageRotVect_.clear();
    imageTransVect_.clear();

    // Acquire the mutices
    std::lock(velodyneMtx_, elevationMapMtx_, frontMtx_, backMtx_, leftMtx_, rightMtx_, voxelMapMtx_);

    auto img_ts = imgTs_;
    // For the images, compute the rotation and translation
    Matrix4d frontTF = T_sensor_gravity__map_ * T_map__front_cam_link_;
    Matrix4d backTF = T_sensor_gravity__map_ * T_map__back_cam_link_;
    Matrix4d leftTF = T_sensor_gravity__map_ * T_map__left_cam_link_;
    Matrix4d rightTF = T_sensor_gravity__map_ * T_map__right_cam_link_;

    // Convert each image to vector of Eigen Matrices for easy pybind conversion
    // std::vector<ColMatrixXf> frontImg, backImg, leftImg, RightImg;

    // convertToMultichannel(frontImage_, leftImg);
    // convertToMultichannel(leftImage_, leftImg);
    // convertToMultichannel(rightImage_, RightImg);
    // convertToMultichannel(backImage_, backImg);
    std::vector<ColMatrixXf> frontImg(frontImage_);
    std::vector<ColMatrixXf> leftImg(leftImage_);
    std::vector<ColMatrixXf> rightImg(rightImage_);
    std::vector<ColMatrixXf> backImg(backImage_);

    imageVect_.insert(imageVect_.end(), { frontImg, leftImg, rightImg, backImg });
    camFrontRec_ = false;
    camBackRec_ = true;

    // Release the Image mutices
    frontMtx_.unlock();
    backMtx_.unlock();
    leftMtx_.unlock();
    rightMtx_.unlock();

    // Put rotation, translation and Intrinsics in vector
    imageRotVect_.insert(imageRotVect_.end(), { frontTF.block<3, 3>(0, 0), leftTF.block<3, 3>(0, 0),
                                                rightTF.block<3, 3>(0, 0), backTF.block<3, 3>(0, 0) });
    imageTransVect_.insert(imageTransVect_.end(), { frontTF.block<3, 1>(0, 3), leftTF.block<3, 1>(0, 3),
                                                    rightTF.block<3, 1>(0, 3), backTF.block<3, 1>(0, 3) });
    kVect_.insert(kVect_.end(), { P_front_.block<3, 3>(0, 0), P_left_.block<3, 3>(0, 0), P_right_.block<3, 3>(0, 0),
                                  P_back_.block<3, 3>(0, 0) });

    // Transform the Raw PCD data to the sensor gravity frame (T-sensor_grav__map * T-map__base_link * raw PCD)
    // For now we are just passing the TF and the Pointcloud in baselink frame
    Matrix4d T_sensor_gravity__baselink = T_sensor_gravity__map_ * T_map__base_link_;

    // std::cout << "PCD SIZE: " << pcd_.rows() << " , "<< pcd_.cols() << std::endl;
    RowMatrixXf pcd_data = pcd_;
    velodyneRec_ = false;
    // Release the PCD Mutex
    velodyneMtx_.unlock();

    RowMatrixXf gvom_data = gvom_;
    // Release the PCD Mutex
    voxelMapMtx_.unlock();

    // For GridMap
    // GridMap eleMap;
    // grid_map::GridMapRosConverter::fromMessage(*ele_msg_, eleMap);

    // // Get the size of the grid map.
    // size_t numRows = eleMap.getSize()(0);
    // size_t numCols = eleMap.getSize()(1);
    // RowMatrixXf eleMapEigen(numRows, numCols);

    size_t numRows = 500;
    size_t numCols = 500;
    RowMatrixXf eleMapEigen = RowMatrixXf::Zero(numRows, numCols);

    // Find the TF (Yaw and x,y,z)
    Matrix4d T_map__grid_map_center = Matrix4d::Identity();
    // T_map__grid_map_center(0, 3) = ele_msg_->info.pose.position.x;
    // T_map__grid_map_center(1, 3) = ele_msg_->info.pose.position.y;
    // T_map__grid_map_center(2, 3) = ele_msg_->info.pose.position.z;

    Matrix4d T_sensor_gravity__grid_map_center = T_sensor_gravity__map_ * T_map__grid_map_center;
    // Convert rotation matrix to yaw-pitch-roll (ZYX) Euler angles
    // double yaw = rotationAsYPR(T_sensor_gravity__grid_map_center.block<3, 3>(0, 0))[0];
    double yaw = 0;

    // {
    //   py::gil_scoped_acquire acquire;
    //   auto ypr = pyHandle_.attr("get_ypr_py")(pyHandle_, T_sensor_gravity__grid_map_center);
    //   py::gil_scoped_release release;
    //   std::cout << "Received Yaw from py" << std::endl;
    //   yaw = pyArrayToEigenVector(ypr)(0,0);
    // }

    // Find Yaw and shift    std::cout << "Yaw is: " << yaw << " | " << yaw_cpp * (180/M_PI) << std::endl;
    // double resolution = ele_msg_->info.resolution;
    double resolution = 0.2;
    // Shift is x,y (Need to reverse order in python for Torch tensor)
    // Eigen::Vector2d shift(T_sensor_gravity__grid_map_center(0, 3) / resolution,
    //                       T_sensor_gravity__grid_map_center(1, 3) / resolution);
    Eigen::Vector2d shift(0, 0);

    // Iterate through the GridMap and populate the Eigen Matrix.
    // for (grid_map::GridMapIterator iterator(eleMap); !iterator.isPastEnd(); ++iterator)
    // {
    //   const grid_map::Index index(*iterator);
    //   // Assign the elevation value to the corresponding location in the Eigen Matrix and add the Z Offset
    //   eleMapEigen(index(0), index(1)) =
    //       eleMap.at("elevation_raw", index) + static_cast<float>(T_sensor_gravity__grid_map_center(2, 3));
    // }

    // Find the TF Matrix (T-sensor_grav__map * T-map__gridmap_center)
    // Filter out to keep only the aux Layers (raw elevation, wheelrisk, reliability)
    // Modify the elevation layers to compensate for Z
    // Compute yaw from the T matrix
    // Pass the yaw, shift and Filtered Grid Map (The rotation can be performed in Python)
    std::cout << "Yaw is: " << yaw << std::endl << "Shift is: " << shift << std::endl;

    // Release the Gridmap Mutex
    elevationMapMtx_.unlock();

    auto time_before_py = std::chrono::high_resolution_clock::now();

    std::vector<std::string> layer_names = {
      "wheel_risk_bev",
      "elevation",
    };

    std::vector<std::string> layer_names_short = {
      "cost",
      "elevation",
    };

    // Map layer pointers
    std::vector<std::unique_ptr<RowMatrixXf>> layer_ptr_micro(layer_names.size());
    std::vector<std::unique_ptr<RowMatrixXf>> layer_ptr_short(layer_names_short.size());

    // Create Gridmap
    grid_map::GridMap gridMapMicro;
    grid_map::GridMap gridMapShort;
    // Call the Python Function for inference
    {
      py::gil_scoped_acquire acquire;
      layer_ptr_micro[0] = std::make_unique<RowMatrixXf>(mapWidthMicro_, mapWidthMicro_);
      layer_ptr_micro[1] = std::make_unique<RowMatrixXf>(mapWidthMicro_, mapWidthMicro_);

      layer_ptr_short[0] = std::make_unique<RowMatrixXf>(mapWidthShort_, mapWidthShort_);
      layer_ptr_short[1] = std::make_unique<RowMatrixXf>(mapWidthShort_, mapWidthShort_);

      pyHandle_.attr("infer_python")(
          pyHandle_, imageVect_, imageRotVect_, imageTransVect_, kVect_, pcd_data, gvom_data,
          T_sensor_gravity__baselink, eleMapEigen, yaw, shift, T_sensor_origin_link__map_, T_sensor_gravity__map_,
          Eigen::Ref<RowMatrixXf>(*layer_ptr_micro[0]), Eigen::Ref<RowMatrixXf>(*layer_ptr_micro[1]),
          Eigen::Ref<RowMatrixXf>(*layer_ptr_short[0]), Eigen::Ref<RowMatrixXf>(*layer_ptr_short[1]));

      // auto predictions = pyArrayToTwoEigenMatrices(py_array_pred);
      py::gil_scoped_release release;
      // std::cout << "EIGEN MAT" << (*layer_ptr[0]) << std::endl;
      auto time_before_gridmap = std::chrono::high_resolution_clock::now();
      auto T_map__sensor_gravity = T_sensor_gravity__map_.inverse();
      Eigen::Vector2d gridmap_position(T_map__sensor_gravity(0, 3), T_map__sensor_gravity(1, 3));

      // gridMapMicro.setFrameId(eleMap.getFrameId());
      gridMapMicro.setFrameId("crl_rzr/map");
      gridMapMicro.setTimestamp(img_ts);
      grid_map::Length length_gridmap_micro(mapWidthMicro_ * resMicro_, mapWidthMicro_ * resMicro_);
      gridMapMicro.setGeometry(length_gridmap_micro, resMicro_, gridmap_position);

      gridMapShort.setFrameId("crl_rzr/map");
      gridMapShort.setTimestamp(img_ts);
      grid_map::Length length_gridmap_short(mapWidthShort_ * resShort_, mapWidthShort_ * resShort_);
      gridMapShort.setGeometry(length_gridmap_short, resShort_, gridmap_position);

      for (size_t l = 0; l < layer_names.size(); ++l)
        gridMapMicro.add(layer_names[l], *layer_ptr_micro[l]);

      for (size_t l = 0; l < layer_names_short.size(); ++l)
        gridMapShort.add(layer_names_short[l], *layer_ptr_short[l]);

      auto time_after_gridmap = std::chrono::high_resolution_clock::now();
      auto duration_postprocessing =
          std::chrono::duration_cast<std::chrono::microseconds>(time_after_gridmap - time_before_gridmap).count();
      std::cout << "Time spent gridmap postprocessing: " << duration_postprocessing << " microseconds" << std::endl;
    }
    auto bef_pub = std::chrono::high_resolution_clock::now();

    grid_map::GridMap outputShortMap;
    grid_map::GridMap outputMicroMap;

    if (!filterChainShort_.update(gridMapShort, outputShortMap))
    {
      std::cout << "EROORR" << std::endl;
      ROS_ERROR("Could not update the grid map filter chain!");
      // return;
    }
    if (!filterChainMicro_.update(gridMapMicro, outputMicroMap))
    {
      std::cout << "EROORR" << std::endl;
      ROS_ERROR("Could not update the grid map filter chain!");
      // return;
    }

    publishMaps(gridMapMicro, outputMicroMap, outputShortMap);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto duration_pub = std::chrono::duration_cast<std::chrono::microseconds>(end - bef_pub).count();
    auto duration_preprocessing = std::chrono::duration_cast<std::chrono::microseconds>(time_before_py - start).count();

    std::cout << "Time spent inside the loop: " << duration << " microseconds" << std::endl;
    std::cout << "Time spent CPP preprocessing: " << duration_preprocessing << " microseconds" << std::endl;
    std::cout << "Time spent for publishing: " << duration_pub << " microseconds" << std::endl;
  }
}

void BevInferenceROS::publishMaps(GridMap& map_micro, GridMap& map_micro_vel, GridMap& map_short)
{
  if (pubGridMapMicro_.getNumSubscribers() > 0)
  {
    auto msg = boost::make_shared<GridMapMsg>();
    grid_map::GridMapRosConverter::toMessage(map_micro, *msg);  // TODO Implement with std::move

    int32_t bytes_estimate = -1;
    pubGridMapMicro_.publish(msg);
    // publish a lightweight message for easy use with rostopic delay
    grid_map_msgs::GridMapInfo info_msg;
    info_msg = msg->info;
    pubGridMapMicroInfo_.publish(info_msg);
  }

  if (pubGridMapMicroVel_.getNumSubscribers() > 0)
  {
    auto msg = boost::make_shared<GridMapMsg>();
    grid_map::GridMapRosConverter::toMessage(map_micro_vel, *msg);  // TODO Implement with std::move

    int32_t bytes_estimate = -1;
    pubGridMapMicroVel_.publish(msg);
    // publish a lightweight message for easy use with rostopic delay
    grid_map_msgs::GridMapInfo info_msg;
    info_msg = msg->info;
    pubGridMapMicroVelInfo_.publish(info_msg);
  }

  if (pubGridMapShort_.getNumSubscribers() > 0)
  {
    auto msg = boost::make_shared<GridMapMsg>();
    grid_map::GridMapRosConverter::toMessage(map_short, *msg);  // TODO Implement with std::move

    int32_t bytes_estimate = -1;
    pubGridMapShort_.publish(msg);
    // publish a lightweight message for easy use with rostopic delay
    grid_map_msgs::GridMapInfo info_msg;
    info_msg = msg->info;
    pubGridMapShortInfo_.publish(info_msg);
  }
}

void BevInferenceROS::elevationMapCb(const grid_map_msgs::GridMapConstPtr& msg)
{
  std::cout << "Raw elevation Map callback" << std::endl;
  // lock Mutex
  std::lock_guard<std::mutex> lock(elevationMapMtx_);
  ele_msg_ = msg;
  gridmapRec_ = true;
}

void BevInferenceROS::gvomCb(const sensor_msgs::PointCloud2ConstPtr& gvom_ptr)
{
  std::cout << "Voxel Map callback" << std::endl;

  // TODO (FOr fusing the GVOM Cloud)

  // Check number of points
  if (gvom_ptr->width < voxelMapMinPts_)
  {
    ROS_WARN("BevInferenceROS: Empty VoxelMap Received !!!");
    return;
  }
  else
  {
    ROS_INFO("\033[36mVoxel\033[0m pts: %d", gvom_ptr->width);
  }

  // Acquire Mutex
  std::lock_guard<std::mutex> lock(voxelMapMtx_);
  // Acquire Mutex

  // Create Eigen matrix from pointcloud2
  auto& num_pts = gvom_ptr->width;

  gvom_.resize(num_pts, 3);
  RowMatrixXf probs_pcl(num_pts, probFieldNames_.size());
  pointCloudToMatrix(gvom_ptr, false, gvom_, probs_pcl);
  gvomRec_ = true;
  // std::cout << "Velodyne CB: " << std::this_thread::get_id() << std::endl;
}

void BevInferenceROS::velodyneCb(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr)
{
  // std::cout << "Velodyne LiDAR callback" << std::endl;

  // Check number of points
  if (cloud_ptr->width < pclMinPts_)
  {
    ROS_WARN("BevInferenceROS: Empty PointCloud Received from Velodyne !!!");
    return;
  }
  else
  {
    ROS_INFO("\033[36mVelodyne\033[0m pts: %d", cloud_ptr->width);
  }

  // Acquire Mutex
  std::lock_guard<std::mutex> lock(velodyneMtx_);
  // Lookup for the TF
  try
  {
    T_map__base_link_ = tf2::transformToEigen(tfBufferV_->lookupTransform(mapFrame_, cloud_ptr->header.frame_id,
                                                                          cloud_ptr->header.stamp, ros::Duration(0.1)))
                            .matrix();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), cloud_ptr->header.frame_id.c_str());
    return;
  }
  // Create Eigen matrix from pointcloud2
  auto& num_pts = cloud_ptr->width;

  pcd_.resize(num_pts, 3);
  // RowMatrixXf xyz_pcl(num_pts, 3);
  RowMatrixXf probs_pcl(num_pts, probFieldNames_.size());
  pointCloudToMatrix(cloud_ptr, false, pcd_, probs_pcl);
  velodyneRec_ = true;
  // std::cout << "Velodyne CB: " << std::this_thread::get_id() << std::endl;
}

void BevInferenceROS::camFrontCb(const sensor_msgs::ImageConstPtr& cam_ptr)
{
  static auto prev_ts = ros::Time(0.0);
  auto ts = ros::Time::now();
  std::cout << "Front Cam callback took (s)" << (ts - prev_ts).toSec() << std::endl;
  prev_ts = ts;

  cv::Mat image = cv_bridge::toCvShare(cam_ptr, cam_ptr->encoding)->image;

  // Change encoding to RGB/RGBA
  if (cam_ptr->encoding == "bgr8")
  {
    cv::cvtColor(image, image, CV_BGR2RGB);
  }
  else if (cam_ptr->encoding == "bgra8")
  {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }

  // Perform image processing (resizing) in-place using OpenCV
  cv::resize(image, image, cv::Size(resizeWidth_, resizeHeight_));

  std::lock_guard<std::mutex> lock(frontMtx_);
  // Lookup For the TF
  try
  {
    T_map__front_cam_link_ =
        tf2::transformToEigen(tfBufferFC_->lookupTransform(mapFrame_, cam_ptr->header.frame_id, cam_ptr->header.stamp,
                                                           ros::Duration(0.1)))
            .matrix();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), cam_ptr->header.frame_id.c_str());
    return;
  }

  // Lookup for the Gravity Aligned Ego Frame
  try
  {
    T_sensor_origin_link__map_ =
        tf2::transformToEigen(
            tfBufferFC_->lookupTransform(mapFrame_, egoFrame_, cam_ptr->header.stamp, ros::Duration(0.1)))
            .matrix()
            .inverse();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), egoFrame_.c_str());
    return;
  }

  Eigen::Matrix4d Temp_matrix;
  getGravityAligned(T_sensor_origin_link__map_, Temp_matrix);
  T_sensor_gravity__map_ = Temp_matrix;
  std::cout << "CPP Matrix from CPP is " << Temp_matrix << std::endl;

  // Store the procestsed image data in a class variable (processed_image_data_)
  // frontImage_ = std::move(image);
  frontImage_.clear();
  convertToMultichannel(image, frontImage_);
  imgTs_ = cam_ptr->header.stamp.toNSec();
  camFrontRec_ = true;
}

void BevInferenceROS::camBackCb(const sensor_msgs::ImageConstPtr& cam_ptr)
{
  std::cout << "Back Cam callback" << std::endl;
  cv::Mat image = cv_bridge::toCvShare(cam_ptr, cam_ptr->encoding)->image;

  // Change encoding to RGB/RGBA
  if (cam_ptr->encoding == "bgr8")
  {
    cv::cvtColor(image, image, CV_BGR2RGB);
  }
  else if (cam_ptr->encoding == "bgra8")
  {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }

  cv::resize(image, image, cv::Size(resizeWidth_, resizeHeight_));

  std::lock_guard<std::mutex> lock(backMtx_);
  // Lookup For the TF
  try
  {
    T_map__back_cam_link_ =
        tf2::transformToEigen(tfBufferBC_->lookupTransform(mapFrame_, cam_ptr->header.frame_id, cam_ptr->header.stamp,
                                                           ros::Duration(0.1)))
            .matrix();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), cam_ptr->header.frame_id.c_str());
    return;
  }

  // backImage_ = std::move(image);
  backImage_.clear();
  convertToMultichannel(image, backImage_);
  camBackRec_ = true;
}

void BevInferenceROS::camLeftCb(const sensor_msgs::ImageConstPtr& cam_ptr)
{
  std::cout << "Left Cam callback" << std::endl;

  cv::Mat image = cv_bridge::toCvShare(cam_ptr, cam_ptr->encoding)->image;

  // Change encoding to RGB/RGBA
  if (cam_ptr->encoding == "bgr8")
  {
    cv::cvtColor(image, image, CV_BGR2RGB);
  }
  else if (cam_ptr->encoding == "bgra8")
  {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }

  cv::resize(image, image, cv::Size(resizeWidth_, resizeHeight_));

  std::lock_guard<std::mutex> lock(leftMtx_);
  // Lookup For the TF
  try
  {
    T_map__left_cam_link_ =
        tf2::transformToEigen(tfBufferLC_->lookupTransform(mapFrame_, cam_ptr->header.frame_id, cam_ptr->header.stamp,
                                                           ros::Duration(0.1)))
            .matrix();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), cam_ptr->header.frame_id.c_str());
    return;
  }

  // leftImage_ = std::move(image);
  leftImage_.clear();
  convertToMultichannel(image, leftImage_);
  camLeftRec_ = true;
  // std::cout << "Left CB: " << std::this_thread::get_id() << std::endl;
}

void BevInferenceROS::camRightCb(const sensor_msgs::ImageConstPtr& cam_ptr)
{
  std::cout << "Right Cam callback" << std::endl;

  cv::Mat image = cv_bridge::toCvShare(cam_ptr, cam_ptr->encoding)->image;

  // Change encoding to RGB/RGBA
  if (cam_ptr->encoding == "bgr8")
  {
    cv::cvtColor(image, image, CV_BGR2RGB);
  }
  else if (cam_ptr->encoding == "bgra8")
  {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }

  cv::resize(image, image, cv::Size(resizeWidth_, resizeHeight_));

  std::lock_guard<std::mutex> lock(rightMtx_);
  // Lookup For the TF
  try
  {
    T_map__right_cam_link_ =
        tf2::transformToEigen(tfBufferRC_->lookupTransform(mapFrame_, cam_ptr->header.frame_id, cam_ptr->header.stamp,
                                                           ros::Duration(0.1)))
            .matrix();
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s\nSKIPPED %s", ex.what(), cam_ptr->header.frame_id.c_str());
    return;
  }

  // rightImage_ = std::move(image);
  rightImage_.clear();
  convertToMultichannel(image, rightImage_);
  camRightRec_ = true;
}

// Following Code Snippet taken from GVOM (voxel_mapper_ros.cpp)
// Convert pointcloud to Eigen matrices (xyz, probs)
void BevInferenceROS::pointCloudToMatrix(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr, bool use_semantics,
                                         RowMatrixXf& xyz, RowMatrixXf& probs)
{
  probs.setZero();  // probabilities are accumulated so initialize to zero

  std::vector<std::string> pcl_labels;
  std::vector<size_t> prob_idx_pcl_to_map;

  // TODO : Some error in the semantics as of now
  // Create a iterator for each pcl field of interest (pclFieldNames_)
  // for (unsigned int i = 0; i < cloud_ptr->fields.size(); ++i)
  // {
  //   if (cloud_ptr->fields[i].name.find("prob") != std::string::npos)
  //   {
  //     pcl_labels.push_back(cloud_ptr->fields[i].name);
  //     if (probFieldMap_.find(cloud_ptr->fields[i].name) != probFieldMap_.end())
  //     {
  //       prob_idx_pcl_to_map.push_back(probFieldMap_.at(cloud_ptr->fields[i].name));
  //     }
  //     else
  //     {
  //       if (useProbOther_)
  //       {
  //         prob_idx_pcl_to_map.push_back(probFieldMap_.at("prob_other"));
  //       }
  //       else
  //       {
  //         prob_idx_pcl_to_map.push_back(-1);
  //       }
  //     }
  //   }
  // }

  // std::cout << "Creating all iterators" << std::endl;
  // //// Create iterators for all fields of pointcloud
  // pclConstItrVec pclItrs = createPointCloud2ConstIterators(cloud_ptr, pcl_labels);
  pclConstItrVec pclItrs;
  pclItrs.reserve(3);

  pclItrs.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "x"));
  pclItrs.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "y"));
  pclItrs.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "z"));

  // Loop though pointcloud points and copy them to
  size_t pt_idx = 0;
  while (pclItrs[0] != pclItrs[0].end())
  {
    // XYZ
    xyz(pt_idx, 0) = static_cast<float>(*pclItrs[0]);
    xyz(pt_idx, 1) = static_cast<float>(*pclItrs[1]);
    xyz(pt_idx, 2) = static_cast<float>(*pclItrs[2]);

    // // Semantic probabilities
    // if (use_semantics) {
    //   size_t offset = 3;  // Offset to start class probability fields
    //   float sum_prob_other = 0;
    //   for (size_t pcl_prob_id = 0; pcl_prob_id < pcl_labels.size(); ++pcl_prob_id)
    //   {
    //     if (prob_idx_pcl_to_map[pcl_prob_id] != -1)
    //     {
    //       probs(pt_idx, prob_idx_pcl_to_map[pcl_prob_id]) += static_cast<float>(*pclItrs[pcl_prob_id + offset]);
    //     }
    //   }
    // }

    // Incrememnt all iterators
    for (auto& itr : pclItrs)
      ++itr;

    // Increment points
    ++pt_idx;
  }
}

// Helper function to create iterators for all fields of pointcloud (const)
pclConstItrVec BevInferenceROS::createPointCloud2ConstIterators(const sensor_msgs::PointCloud2ConstPtr& cloud_ptr,
                                                                const std::vector<std::string>& pcl_labels)
{
  pclConstItrVec itr;
  itr.reserve(pcl_labels.size() + 3);

  itr.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "x"));
  itr.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "y"));
  itr.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, "z"));

  // Create a iterator for each pcl field of interest (pcl_labels)
  for (auto label : pcl_labels)
  {
    itr.push_back(sensor_msgs::PointCloud2ConstIterator<float>(*cloud_ptr, label));
  }

  return itr;
}