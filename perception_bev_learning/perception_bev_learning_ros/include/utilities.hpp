
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pybind11_catkin/pybind11/eigen.h>
#include <pybind11_catkin/pybind11/embed.h>
#include <pybind11_catkin/pybind11/numpy.h>
#include <pybind11_catkin/pybind11/stl.h>

namespace py = pybind11;

Eigen::Vector3d rotationAsYPR2(const Eigen::Matrix3d& RotMat)
{
  // This is a generic function and can be generalized to other sequences
  // https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
  // See Wikipedia https://en.wikipedia.org/wiki/Euler_angles for direct solution -> Z1-Y2-X3 (Tait-Byran)
  // Supports only extrinsic rotation as of now
  double eps = 1e-7;
  // COnvert Rotation Matrix to Quaternion
  Eigen::Quaterniond _quat(RotMat);
  Eigen::Vector3d ypr;
  double a = _quat.w() - _quat.y();
  double b = _quat.z() + _quat.x();
  double c = _quat.y() + _quat.w();
  double d = -_quat.x() - _quat.z();
  double n2 = a * a + b * b + c * c + d * d;

  ypr[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

  bool safe1 = std::abs(ypr[1]) > eps;
  bool safe2 = std::abs(ypr[1] - M_PI) > eps;
  bool safe = safe1 && safe2;

  if (safe)
  {
    double half_sum = std::atan2(b, a);
    double half_diff = std::atan2(-d, c);

    ypr[0] = half_sum + half_diff;
    ypr[2] = half_sum - half_diff;
  }
  else
  {
    if (!safe)
    {
      ypr[2] = 0;
    }
    if (!safe1)
    {
      double half_sum = std::atan2(b, a);
      ypr[0] = 2 * half_sum;
    }
    if (!safe2)
    {
      double half_diff = std::atan2(-d, c);
      ypr[0] = 2 * half_diff;
    }
  }

  for (size_t i; i < 3; ++i)
  {
    if (ypr[i] < -M_PI)
    {
      ypr[i] += 2 * M_PI;
    }
    else if (ypr[i] > M_PI)
    {
      ypr[i] -= 2 * M_PI;
    }
  }

  // For Tait-Byran Angles
  ypr[2] *= -1;
  ypr[1] -= M_PI / 2;

  return ypr;
}

Eigen::Vector3d rotationAsYPR(const Eigen::Matrix3d& RotMat)
{
  // See Wikipedia https://en.wikipedia.org/wiki/Euler_angles for direct solution -> X1-Y2-Z3 (Tait-Byran)
  Eigen::Vector3d ypr;
  ypr[2] = std::atan2(-RotMat(1, 2), RotMat(2, 2));
  ypr[1] = std::asin(RotMat(0, 2));
  ypr[0] = std::atan2(-RotMat(0, 1), RotMat(0, 0));
  return ypr;
}

Eigen::Matrix3d yprToRotation(const Eigen::Vector3d& ypr)
{
  // https://en.wikipedia.org/wiki/Euler_angles
  // For Now, the SOlution X1-Y2-Z3 works but with jumbled Rot Mat entries
  // TODO Figure out and understand this
  Eigen::Matrix3d rotMat;

  // rotMat(0,0) = std::cos(ypr[0])*std::cos(ypr[1]);
  // rotMat(0,1) = std::cos(ypr[0])*std::sin(ypr[1])*std::sin(ypr[2]) - std::cos(ypr[2])*std::sin(ypr[0]);
  // rotMat(0,2) = std::sin(ypr[0])*std::sin(ypr[2]) + std::cos(ypr[0])*std::cos(ypr[2])*std::sin(ypr[1]);
  // rotMat(1,0) = std::cos(ypr[1])*std::sin(ypr[0]);
  // rotMat(1,1) = std::cos(ypr[0])*std::cos(ypr[2]) + std::sin(ypr[0])*std::sin(ypr[1])*std::sin(ypr[2]);
  // rotMat(1,2) = std::cos(ypr[2])*std::sin(ypr[0])*std::sin(ypr[1]) - std::cos(ypr[0])*std::sin(ypr[2]);
  // rotMat(2,0) = -std::sin(ypr[1]);
  // rotMat(2,1) = std::cos(ypr[1])*std::sin(ypr[2]);
  // rotMat(2,2) = std::cos(ypr[1])*std::cos(ypr[2]);

  // rotMat(0,0) = std::cos(ypr[1])*std::cos(ypr[2]);
  // rotMat(0,1) = -std::cos(ypr[1])*std::sin(ypr[2]);
  // rotMat(0,2) = std::sin(ypr[1]);
  // rotMat(1,0) = std::cos(ypr[0])*std::sin(ypr[2]) + std::cos(ypr[2])*std::sin(ypr[0])*std::sin(ypr[1]);
  // rotMat(1,1) = std::cos(ypr[0])*std::cos(ypr[2]) - std::sin(ypr[0])*std::sin(ypr[1])*std::sin(ypr[2]);
  // rotMat(1,2) = -std::cos(ypr[1])*std::sin(ypr[0]);
  // rotMat(2,0) = std::sin(ypr[0])*std::sin(ypr[2]) - std::cos(ypr[0])*std::cos(ypr[2])*std::sin(ypr[1]);
  // rotMat(2,1) = std::cos(ypr[2])*std::sin(ypr[0]) + std::cos(ypr[0])*std::sin(ypr[1])*std::sin(ypr[2]);
  // rotMat(2,2) = std::cos(ypr[0])*std::cos(ypr[1]);

  rotMat(2, 2) = std::cos(ypr[1]) * std::cos(ypr[2]);
  rotMat(1, 2) = -std::cos(ypr[1]) * std::sin(ypr[2]);
  rotMat(0, 2) = std::sin(ypr[1]);
  rotMat(2, 1) = std::cos(ypr[0]) * std::sin(ypr[2]) + std::cos(ypr[2]) * std::sin(ypr[0]) * std::sin(ypr[1]);
  rotMat(1, 1) = std::cos(ypr[0]) * std::cos(ypr[2]) - std::sin(ypr[0]) * std::sin(ypr[1]) * std::sin(ypr[2]);
  rotMat(0, 1) = -std::cos(ypr[1]) * std::sin(ypr[0]);
  rotMat(2, 0) = std::sin(ypr[0]) * std::sin(ypr[2]) - std::cos(ypr[0]) * std::cos(ypr[2]) * std::sin(ypr[1]);
  rotMat(1, 0) = std::cos(ypr[2]) * std::sin(ypr[0]) + std::cos(ypr[0]) * std::sin(ypr[1]) * std::sin(ypr[2]);
  rotMat(0, 0) = std::cos(ypr[0]) * std::cos(ypr[1]);

  return rotMat;
}

void getGravityAligned(const Eigen::Matrix4d& T_f_map, Eigen::Matrix4d& T_gravity_aligned)
{
  // Copy the matrix
  T_gravity_aligned = T_f_map;

  // Convert rotation matrix to yaw-pitch-roll (ZYX) Euler angles
  Eigen::Vector3d ypr = rotationAsYPR(T_f_map.block<3, 3>(0, 0));

  std::cout << "YPR CPP is: " << ypr << std::endl;
  // Set yaw to zero
  ypr[0] = 0;

  // Convert back Euler angles to rotation matrix
  Eigen::Matrix4d T_delta = Eigen::Matrix4d::Identity();
  //   Eigen::Quaterniond correctedQuaternion = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
  //                                            Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
  //                                            Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());

  //   // Convert the corrected quaternion back to a rotation matrix
  //   T_delta.block<3, 3>(0, 0) = correctedQuaternion.toRotationMatrix();
  T_delta.block<3, 3>(0, 0) = yprToRotation(ypr);
  // Apply the correction
  T_gravity_aligned = T_delta.inverse() * T_gravity_aligned;
}

Eigen::MatrixXd pyArrayToEigenMatrix(const py::array_t<double>& array)
{
  // Get buffer info from the NumPy array
  py::buffer_info bufInfo = array.request();

  // Create an Eigen::Map to the NumPy array data
  Eigen::Map<Eigen::MatrixXd> eigenMap(static_cast<double*>(bufInfo.ptr),  // Pointer to the data
                                       bufInfo.shape[0],                   // Number of rows
                                       bufInfo.shape[1]                    // Number of columns
  );

  // Return the Eigen matrix
  return eigenMap;
}

Eigen::MatrixXd pyArrayToEigenVector(const py::array_t<double>& array)
{
  // Get buffer info from the NumPy array
  py::buffer_info bufInfo = array.request();

  // Create an Eigen::Map to the NumPy array data
  Eigen::Map<Eigen::VectorXd> eigenMap(static_cast<double*>(bufInfo.ptr),  // Pointer to the data
                                       bufInfo.shape[0], 1                 // Number of rows
  );

  // Return the Eigen matrix
  return eigenMap;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> pyArrayToTwoEigenMatrices(py::array_t<double> numpy_array)
{
  auto buffer_info = numpy_array.request();
  double* ptr = static_cast<double*>(buffer_info.ptr);

  int rows = buffer_info.shape[1];  // Assuming shape[0] is 2, as mentioned in your question
  int cols = buffer_info.shape[2];

  // Map the data to Eigen matrices
  Eigen::Map<Eigen::MatrixXd> matrix1(ptr, rows, cols);
  Eigen::Map<Eigen::MatrixXd> matrix2(ptr + rows * cols, rows, cols);

  return std::make_pair(matrix1, matrix2);
}
