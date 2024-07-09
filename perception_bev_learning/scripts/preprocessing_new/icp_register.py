import copy

import numpy as np
import open3d as o3d
from scipy.interpolate import griddata


# Step 1: Convert elevation gridmaps to point clouds
def gridmap_to_pointcloud(gridmap, multiplier=0.2):
    indices = np.where(~np.isnan(gridmap))
    points = np.column_stack(
        (
            multiplier * indices[0] - multiplier * gridmap.shape[0] // 2,
            multiplier * indices[1] - multiplier * gridmap.shape[1] // 2,
            gridmap[indices],
        )
    )
    # print(points.shape)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def pointcloud_to_gridmap(pointcloud, multiplier=0.2, grid_shape=(110, 110)):
    # Get the points and colors from the point cloud
    points = np.asarray(pointcloud.points)
    values = points[:, 2]  # Assuming z-values represent the values in the gridmap

    # Normalize coordinates to match the grid indices
    normalized_coords = (
        (points[:, :2] / multiplier) + np.array(grid_shape) / 2
    ).astype(int)
    # Create an empty grid map
    gridmap = np.full(grid_shape, np.nan, dtype=float)
    # Assign values to the corresponding grid cells
    gridmap[normalized_coords[:, 0], normalized_coords[:, 1]] = values
    # Find indices of missing values (NaN)
    missing_indices = np.isnan(gridmap)

    # Create a meshgrid for the entire grid map
    x, y = np.meshgrid(np.arange(grid_shape[1]), np.arange(grid_shape[0]))

    # Interpolate missing values using griddata
    interpolated_values = griddata(
        (normalized_coords[:, 1], normalized_coords[:, 0]),
        values,
        (x, y),
    )
    # Replace missing values with interpolated values
    gridmap[missing_indices] = interpolated_values[missing_indices]

    return gridmap


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size, return_features=False):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    if return_features:
        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        return pcd_down, pcd_fpfh
    else:
        return pcd_down


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def execute_icp(source_down, target_down, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
    )
    return result


def main():
    print(":: Load two mesh.")
    # target_mesh = o3d.io.read_triangle_mesh('bunny.ply')
    # source_mesh = copy.deepcopy(target_mesh)
    # source_mesh.rotate(source_mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)), center=(0, 0, 0))
    # source_mesh.translate((0, 0.05, 0))
    target = o3d.io.read_point_cloud("pointcloud1.pcd")
    source = o3d.io.read_point_cloud("pointcloud2.pcd")

    draw_registration_result(target, source, np.identity(4))

    # print(":: Sample mesh to point cloud")
    # target = target_mesh.sample_points_uniformly(1000)
    # source = source_mesh.sample_points_uniformly(1000)
    # draw_registration_result(source, target, np.identity(4))
    voxel_size = 5
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_ransac = execute_icp(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    draw_registration_result(source, target, result_ransac.transformation)


if __name__ == "__main__":
    main()
