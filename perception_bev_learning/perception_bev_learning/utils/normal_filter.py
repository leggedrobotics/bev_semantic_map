import torch
from torchvision.transforms.functional import affine
import torchvision


def normal_filter_torch(elevation, resolution):
    """
    Simple normal calculation
    elevation torch.tensor shape:=(H,W): nan values indicate invalid height
    Returns normal map with: nx,ny,nz - nan values indicate not compute normals
    """
    elevation = elevation[None]

    kernel = torchvision.transforms.GaussianBlur((5, 5), sigma=3)
    tmp = kernel(elevation.clone())
    elevation_shift_x = affine(tmp[:, :], 0, [1, 0], 1, [0, 0], fill=torch.nan)
    elevation_shift_y = affine(tmp[:, :], 0, [0, 1], 1, [0, 0], fill=torch.nan)
    dzdx = elevation_shift_x - tmp
    dzdy = elevation_shift_y - tmp
    nx = -dzdy / resolution
    ny = -dzdx / resolution
    nz = 1.0
    norm = torch.sqrt((nx * nx) + (ny * ny) + 1)

    return torch.stack([nx / norm, ny / norm, nz / norm])


if __name__ == "__main__":
    d = "cuda"
    # ele_map = torch.rand((512, 512), device=d)
    ele_map = torch.linspace(0, 100, 512)[None].repeat(512, 1)
    mask_map = torch.ones((512, 512), device=d, dtype=torch.bool)
    n_elements = 512 * 512
    res = normal_filter_torch(ele_map, 0.1)
    print(res)
