from perception_bev_learning.utils import load, inv
import torch
from perception_bev_learning.utils import denormalize_img
import numpy as np
from PIL import Image, ImageDraw

# data = load()

# camera_info, imgs, pcd_new = data["camera_info"], data["imgs"], data["pcd_new"]


def simple_visu(camera_info, imgs, pcd_new):
    rots, trans, intrins, post_rots, post_trans = (
        camera_info["rots"],
        camera_info["trans"],
        camera_info["intrins"],
        camera_info["post_rots"],
        camera_info["post_trans"],
    )
    BS, CAMS, C, H, W = imgs.shape
    # imgs = imgs.reshape(-1,C,H,W)

    H_sensor_gravity__cam = torch.eye(4, 4, device=imgs.device)[None, None].repeat(BS, CAMS, 1, 1)
    H_sensor_gravity__cam[:, :, :3, :3] = rots
    H_sensor_gravity__cam[:, :, :3, 3] = trans

    # H_cam__sensor_gravity = torch.stack([inv(t) for t in torch.unbind(H_sensor_gravity__cam)])
    # intrins = intrins.reshape(-1,3,3)
    # post_rots = post_rots.reshape(-1,3,3)
    # post_trans = post_trans.reshape(-1,3)
    start = 0
    b = 0

    imgs_ls = []

    for b in range(0, BS):
        pcd_idx = b
        stop = start + pcd_new["batch"][b]
        points = pcd_new["points"][start:stop]
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        start += pcd_new["batch"][b]

        for cam in range(CAMS):
            H_cam__sensor_gravity = inv(H_sensor_gravity__cam[b, cam])

            dist = torch.linalg.norm((H_cam__sensor_gravity[:3] @ points.clone().T)[:3, :], dim=0)

            uv = (intrins[b, cam] @ H_cam__sensor_gravity[:3] @ points.clone().T).T
            mask_in_front_of_camera = uv[:, 2] > 0
            uv = uv[:, :] / uv[:, 2:]

            uv = (post_rots[b, cam] @ uv.T).T + post_trans[b, cam]
            # uv = (post_rots[b,cam].T @ uv.T).T + post_trans[b,cam]
            # uv = (post_rots[b,cam] @ uv.T).T - post_trans[b,cam]
            # uv = (post_rots[b,cam].T @ uv.T).T - post_trans[b,cam]

            # uv = (post_rots[b,cam] @ (uv+post_trans[b,cam]).T).T
            # uv = (post_rots[b,cam].T @ (uv+post_trans[b,cam]).T).T
            # uv = (post_rots[b,cam] @ (uv-post_trans[b,cam]).T).T
            # uv = (post_rots[b,cam].T @ (uv-post_trans[b,cam]).T).T

            mask_inside_frustrum = (uv[:, 0] >= 0) * (uv[:, 1] >= 0) * (uv[:, 0] < W) * (uv[:, 1] < H)
            m = mask_inside_frustrum * mask_in_front_of_camera
            dist = dist[m]
            uv = uv[m]

            dist = dist / dist.max()

            background = denormalize_img(imgs[b, cam])
            background.convert("RGBA")
            foreground = np.array(background).copy()
            foreground[:, :, :] = 0
            foreground = Image.fromarray(foreground).convert("RGBA")
            foreground.putalpha(0)
            draw = ImageDraw.Draw(foreground)
            w = 1

            uv = uv.cpu().tolist()
            for i in range(len(uv)):
                try:
                    draw.arc(
                        [(uv[i][0], uv[i][1]), (uv[i][0] + w, uv[i][1] + w)],
                        start=0,
                        end=360,
                        fill=(int(255 * float(dist[i])), int(255 - (255 * float(dist[i]))), 0),
                        width=w,
                    )
                except Exception as e:
                    print(e)
                    pass

            background.paste(foreground, (0, 0), foreground)
            imgs_ls.append(torch.from_numpy(np.array(background)).permute(2, 0, 1))

    from torchvision.utils import make_grid

    grid = make_grid(imgs_ls, nrow=CAMS)
    Image.fromarray(np.uint8(grid.permute(1, 2, 0).numpy())).show()
