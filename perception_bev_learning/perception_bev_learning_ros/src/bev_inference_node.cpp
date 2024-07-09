#include "bev_inference_ros.hpp"
#include <signal.h>

// Signal handler function
void sigintHandler(int sig)
{
  // Handle Ctrl+C here, for example, shut down your ROS node
  ros::shutdown();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "bev_inference_node");
  ros::NodeHandle nh, pnh("~");

  // // Set up Ctrl+C signal handler
  // signal(SIGINT, sigintHandler);

  py::scoped_interpreter guard{};  // start the interpreter and keep it alive

  // Initialize the bev inference ros class
  BevInferenceROS bevinf(nh, pnh);

  if (!bevinf.init())
  {
    ROS_ERROR("BEV Inference Initialization Failed");
    return 0;
  }

  // release GIL
  py::gil_scoped_release release;

  // ROS Spin
  ros::AsyncSpinner spinner(6);  // Use n threads
  spinner.start();
  ros::waitForShutdown();
}