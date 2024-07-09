import numpy as np
from pathlib import Path
from moviepy.editor import ImageSequenceClip

_p_visu = "/media/jonfrey/Data2/bev_traversability/results_tacc/first_training_run"
taged_by_front_camera_idx = True
add = 7 * int(taged_by_front_camera_idx)
keys = np.unique(np.array([str(s)[: -(10 + add)] for s in Path(_p_visu).rglob("*.png")])).tolist()
keys = [k.split("/")[-1] for k in keys]

video_path = Path(_p_visu).parent.joinpath("video")
video_path.mkdir(exist_ok=True)

for k in keys:
    imgs = [str(s) for s in Path(_p_visu).rglob(f"{k}*.png")]
    imgs.sort(key=lambda x: int(x[-(10 + add) : -(4 + add)]))
    if len(imgs) != 0:
        clip = ImageSequenceClip(imgs, fps=15)
        clip.write_videofile(str(video_path.joinpath(imgs[0].split("/")[-1][: -(4 + 7 + add)] + ".mp4")), verbose=True)
