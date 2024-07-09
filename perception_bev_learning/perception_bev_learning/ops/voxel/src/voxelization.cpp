#include <torch/extension.h>
#include "voxelization.h"
#include <cstdint>
#include <torch/script.h>

namespace voxelization
{
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("hard_voxelize", &hard_voxelize, "hard voxelize");
  m.def("dynamic_voxelize", &dynamic_voxelize, "dynamic voxelization");
  m.def("dynamic_point_to_voxel_forward", &dynamic_point_to_voxel_forward, "dynamic point to voxel forward");
  m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward, "dynamic point to voxel backward");
}

TORCH_LIBRARY(voxel_ops, m)
{
  m.def("hard_voxelize_ts", hard_voxelize_ts);
}
}  // namespace voxelization
