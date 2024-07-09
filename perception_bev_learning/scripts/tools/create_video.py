import os
from perception_bev_learning import BEV_ROOT_DIR
from perception_bev_learning.utils import load_env

di = load_env()
p = os.path.join(di["result_dir"], "debug/debug", "visu")
os.system(
    f"""cd {p} &&
    ffmpeg -r 10 -f image2 -s 1920x1080 -i 0_train_dashboard_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p -y 00_test.mp4
    """
)
