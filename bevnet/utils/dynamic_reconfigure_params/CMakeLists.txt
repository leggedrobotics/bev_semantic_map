cmake_minimum_required(VERSION 3.0.2)
project(dynamic_reconfigure_params)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  dynamic_reconfigure
)

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/Tutorials.cfg
)

catkin_package(
)

catkin_install_python(
  PROGRAMS
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(
  DIRECTORY src
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
