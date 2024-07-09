from perception_bev_learning import BEV_ROOT_DIR
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
from pathlib import Path


def plot_gpx(gpx_file, color):
    gpx = gpxpy.parse(open(gpx_file))

    latitudes = []
    longitudes = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)

    plt.scatter(longitudes, latitudes, marker="o", s=5, c=color, alpha=0.7, label=gpx_file.stem)


if __name__ == "__main__":
    gpx_dir = Path(BEV_ROOT_DIR, "assets/gps")
    gpx_files = list(gpx_dir.glob("*.gpx"))

    h5py_dir = Path("/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/add_inpainted_current")
    h5py_files = list(h5py_dir.glob("*.h5py"))

    plt.figure(figsize=(10, 6))
    k = 0
    for idx, gpx_file in enumerate(gpx_files):
        keep = False
        for h5py_file in h5py_files:
            t = str(gpx_file).split("/")[-1][:-4]
            if str(h5py_file).find(t) != -1:
                keep = True
                k += 1

        if keep:
            colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow", "black", "grey"]
            color = colors[k % len(colors)]
            plot_gpx(gpx_file, color)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GPX Scatter Plot")
    plt.grid(True)
    plt.legend()
    plt.show()
