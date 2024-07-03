cmake_minimum_required(VERSION 3.0.2)
project(perception_bev_learning_ros)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

set(CATKIN_DEPS
  std_msgs
  grid_map_ros
  grid_map_msgs
  nav_msgs
  ros_numpy
  roscpp
  rospy
  sensor_msgs
  tf
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  std_srvs
  nodelet
  pybind11_catkin
  pcl_conversions
  pcl_ros
  message_filters
  dynamic_reconfigure
  image_transport
  cv_bridge
)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})


find_package(OpenCV 4.0 REQUIRED)
if(NOT OpenCV_FOUND)
  find_package(OpenCV_FOUND)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.0 not found.")
  endif()
endif()

find_package(Eigen3 REQUIRED)
message("Eigen Version:" ${EIGEN3_VERSION_STRING})
message("Eigen Path:" ${Eigen3_DIR})


catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   dynamic_reconfigure/Base.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS ${CATKIN_DEPS}
  DEPENDS EIGEN3
  # LIBRARIES
)

###########
## Build ##
###########


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${PYTHON_INCLUDE_DIRS} 
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

## Add library
add_library(bev_inference_ros  src/bev_inference_ros.cpp)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(bev_inference_node src/bev_inference_node.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(bev_inference_ros ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(bev_inference_node bev_inference_ros ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/hello2.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
# install(TARGETS fast_node
#   LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(TARGETS bev_inference_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)