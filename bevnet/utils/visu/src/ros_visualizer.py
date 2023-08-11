import rospy
from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np

class NumpyToGridmapVisualizer:
    def __init__(self, init_node=True):
        rospy.init_node("np_to_gridmap_visualizer", anonymous=False)
        self.pub_gridmap = rospy.Publisher("gridmap", GridMap, queue_size=1)

    def gridmap_arr(self, arr, res, layers, reference_frame="map", publish=True, x=0, y=0):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(arr.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = arr[i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = rospy.Time.now()
        info.resolution = res
        info.length_x = size_x * res
        info.length_y = size_y * res
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y
        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data)
        if publish:
            print("publishing")
            self.pub_gridmap.publish(gm_msg)
        return gm_msg


if __name__ == "__main__":
    vis = NumpyToGridmapVisualizer()

    shape = (1, 128, 128)
    arr = np.random.rand(*shape) * 3

    # Threshold arr if bigger than 1
    arr[arr > 1] = 1
    arr[arr <= 0] = 0

    res = 0.1
    layers = ["l1"]

    while not rospy.is_shutdown():
        vis.gridmap_arr(arr, res, layers, x=0, y=0)
        rospy.sleep(1)
