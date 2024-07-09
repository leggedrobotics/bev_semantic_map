from perception_bev_learning.visu import paper_colors_rgb_f, paper_colors_rgb_u8, get_img_from_fig
from perception_bev_learning.utils import denormalize_img
from perception_bev_learning import BEV_ROOT_DIR
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import open3d as o3d
import torch
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
from typing import List, Optional


def show(img, title=None):
    if title is None:
        img = Image.fromarray(img)
        img.show(title=title)
    else:
        fig = plt.figure()
        # Display the image using imshow
        plt.imshow(img)

        # Set the title for the plot
        plt.title(title)

        # Hide the axis values and show the plot
        plt.axis("off")
        img = get_img_from_fig(fig)
        img.show(title=title)


def visu_frustrum(points: torch.Tensor, grid_map_elevation: torch.Tensor, res=0.2, N=3):
    x_ = (
        torch.arange(-(grid_map_elevation.shape[0] / 2), (grid_map_elevation.shape[0] / 2), 1, device=points.device)
        * res
    )
    y_ = (
        torch.arange(-(grid_map_elevation.shape[1] / 2), (grid_map_elevation.shape[1] / 2), 1, device=points.device)
        * res
    )
    xv, yv = torch.meshgrid(x_, y_, indexing="ij")
    points_grid = torch.stack([xv, yv, grid_map_elevation], dim=2).reshape(-1, 3)
    points_grid = points_grid[points_grid[:, 2] != 0]
    points_grid = points_grid[~torch.isnan(points_grid[:, 2])]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_grid.cpu().numpy())

    co = [torch.tensor(v, device=points.device) for v in paper_colors_rgb_f.values()]
    pcd = o3d.geometry.PointCloud()
    vis = points[0].reshape(-1, 3).clone()
    col = points[0].reshape(-1, 3).clone()
    delt = int(vis.shape[0] / N)
    for i in range(N):
        col[int(i * delt) : int((i + 1) * delt)] = co[i]
    pcd.points = o3d.utility.Vector3dVector(vis.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(col.cpu().numpy())
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame, pcd2])


def get_example_gridmap(target):
    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            target[0, i, j] = min(((i - target.shape[1] / 2) ** 2 + (j - target.shape[2] / 2) ** 2) ** 0.5, 40)

    target /= target.max()

    return target


def visu_binary_maps(binary_maps: List[torch.tensor], single_plots: bool = False):
    fused_img = np.zeros((binary_maps[0].shape[0], binary_maps[0].shape[1], 3), dtype=np.uint8)
    col = [np.array(v) for v in paper_colors_rgb_u8.values()]
    single_imgs = []
    for j, bm in enumerate(binary_maps):
        single_imgs.append(np.zeros((bm.shape[0], bm.shape[1], 3), dtype=np.uint8))
        bm = bm.cpu().numpy()
        single_imgs[-1][bm] = col[j]
        fused_img[bm] = col[j]

    imgs = single_imgs + [fused_img]
    if len(imgs) < 17:
        nrows_ncols = (4, 4)
    if len(imgs) < 10:
        nrows_ncols = (3, 3)
    if len(imgs) < 7:
        nrows_ncols = (2, 3)
    if len(imgs) < 5:
        nrows_ncols = (2, 2)
    if len(imgs) < 3:
        nrows_ncols = (1, 2)

    fig = plt.figure(figsize=(nrows_ncols[1] * 4.0, nrows_ncols[0] * 4.0))

    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for j, im in enumerate(imgs):
        # Iterating over the grid returns the Axes.
        ax = grid[j]
        ax.imshow(im)

    img = get_img_from_fig(fig)
    img.show()
    plt.close()
