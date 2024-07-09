import os
from pathlib import Path
import shutil

target = "/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2"

# Removing processed
# processed_folders =  [str(s) for s in Path(target).rglob(f"*/processed") if os.path.isdir(str(s))]
# for f in processed_folders:
#     shutil.rmtree(f)
# Removing bags

result_folders = [os.path.join(target, o) for o in os.listdir(target) if os.path.isdir(os.path.join(target, o))]
for f in result_folders:
    for j in [
        "crl_rzr_bev_trav_",
        "crl_rzr_bev_velodyne_",
        "crl_rzr_bev_odometry_",
        "crl_rzr_bev_color_",
        "crl_rzr_bev_tf_",
        "merged_tf",
    ]:
        p = [str(s) for s in Path(f).rglob(f"{j}*")]
        for k in p:
            os.system(f"rm {k}")
