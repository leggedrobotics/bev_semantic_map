from torch import Tensor
from typing import List, Optional, Tuple
import torch
import cupy as cp
import string
from torch.utils.dlpack import from_dlpack
from PIL import Image


# Method copied from KORNIA TODO check license
def get_convex_edges(polygon: Tensor, h: int, w: int) -> Tuple[Tensor, Tensor]:
    r"""Gets the left and right edges of a polygon for each y-coordinate y \in [0, h)
    Args:
        polygons: represents polygons to draw in BxNx2
            N is the number of points
            2 is (x, y).
        h: bottom most coordinate (top coordinate is assumed to be 0)
        w: right most coordinate (left coordinate is assumed to be 0)
    Returns:
        The left and right edges of the polygon of shape (B,B).
    """
    dtype = polygon.dtype

    # Check if polygons are in loop closed format, if not -> make it so
    if not torch.allclose(polygon[..., -1, :], polygon[..., 0, :]):
        polygon = torch.cat((polygon, polygon[..., :1, :]), dim=-2)  # (B, N+1, 2)

    # Partition points into edges
    x_start, y_start = polygon[..., :-1, 0], polygon[..., :-1, 1]
    x_end, y_end = polygon[..., 1:, 0], polygon[..., 1:, 1]

    # Create scanlines, edge dx/dy, and produce x values
    ys = torch.arange(h, device=polygon.device, dtype=dtype)
    dx = ((x_end - x_start) / (y_end - y_start + 1e-12)).clamp(-w, w)
    xs = (ys[..., :, None] - y_start[..., None, :]) * dx[..., None, :] + x_start[
        ..., None, :
    ]

    # Only count edge in their active regions (i.e between the vertices)
    valid_edges = (y_start[..., None, :] <= ys[..., :, None]).logical_and(
        ys[..., :, None] <= y_end[..., None, :]
    )
    valid_edges |= (y_start[..., None, :] >= ys[..., :, None]).logical_and(
        ys[..., :, None] >= y_end[..., None, :]
    )
    x_left_edges = xs.clone()
    x_left_edges[~valid_edges] = w
    x_right_edges = xs.clone()
    x_right_edges[~valid_edges] = -1

    # Find smallest and largest x values for the valid edges
    x_left = x_left_edges.min(dim=-1).values
    x_right = x_right_edges.max(dim=-1).values
    return x_left, x_right


def polygon_drawing_kernel(width, height):
    polygon_drawing_kernel = cp.ElementwiseKernel(
        in_params="raw U x_left, raw U x_right, raw B colors, raw U nr_polygons",
        out_params="raw B image_out",
        preamble=string.Template("").substitute(),
        operation=string.Template(
            """
            // i == nr_polygon
            // image_out # C, H ,W
            // colors # nr_polygons, C

            int layer = ${width} * ${height};
            
            for (int height_ = 0; height_ <  ${height}; height_++){
                
                int left_right_idx = i * ${height} + height_;
                
                for (int width_  = x_left[left_right_idx]; width_ < x_right[left_right_idx]; width_++) {
                    
                    int height_width_idx = height_ * ${width} + width_;
                    int r_idx = layer * 0 + height_width_idx;
                    int g_idx = layer * 1 + height_width_idx;
                    int b_idx = layer * 2 + height_width_idx;
                    int a_idx = layer * 3 + height_width_idx;
                    
                    if ( width_ == x_left[left_right_idx] || width_ == x_right[left_right_idx] - 1){
                        image_out[ r_idx ] = 0; // R   
                        image_out[ g_idx ] = 0; // G
                        image_out[ b_idx ] = 0; // B
                        image_out[ a_idx ] = 254; // A
                    }else{
                        image_out[ r_idx ] = colors[ i*4 + 0 ]; // R   
                        image_out[ g_idx ] = colors[ i*4 + 1 ]; // G
                        image_out[ b_idx ] = colors[ i*4 + 2 ]; // B
                        image_out[ a_idx ] = colors[ i*4 + 3 ]; // A
                    }
                    
                    height_width_idx = height_ * ${width} + x_left[left_right_idx];
                }
            }
            """
        ).substitute(height=height, width=width),
        name="polygon_drawing_kernel",
    )
    return polygon_drawing_kernel


class DrawPolygonHelper:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.polygon_drawing_kernel = polygon_drawing_kernel(width, height)

    def draw_polygons(self, polygons, colors):
        """_summary_

        Args:
            polygons (torch.long): Nx4x2
            colors (torch.uint8): N,4

        Returns:
            _type_: PIL Image
        """
        device = polygons.device
        x_left, x_right = get_convex_edges(
            polygons.type(torch.float32), self.height, self.width
        )
        x_left = x_left.clip(-1, self.width - 1)
        x_left = x_left.type(torch.long).cpu().numpy()
        x_right = x_right.clip(-1, self.width)
        x_right = x_right.type(torch.long).cpu().numpy()
        colors = colors.cpu().numpy()

        image_out = cp.asarray(
            torch.zeros((4, self.height, self.width), dtype=torch.uint8, device=device)
        )

        self.polygon_drawing_kernel(
            cp.asarray(x_left),
            cp.asarray(x_right),
            cp.asarray(colors),
            int(colors.shape[0]),
            image_out,
            size=(x_left.shape[0]),
        )
        return torch.from_numpy(cp.asnumpy(image_out)).to(device)

        # return from_dlpack(image_out.toDlpack())


if __name__ == "__main__":
    import numpy as np

    device = "cuda"
    H, W = 200, 400

    dph = DrawPolygonHelper(H, W)

    # each row is a polygon with 4 points
    pol = torch.tensor(
        [
            [[0, 0], [0, 20], [20, 20], [20, 0]],
            [[50, 50], [50, 70], [70, 70], [70, 50]],
            [[100, 100], [100, 120], [120, 120], [120, 100]],
        ],
        dtype=torch.long,
        device=device,
    )
    colors = torch.tensor(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
        dtype=torch.uint8,
        device=device,
    )
    foreground = dph.draw_polygons(pol, colors)
    foreground = Image.fromarray(foreground.permute(1, 2, 0).cpu().numpy())
    foreground.show()
