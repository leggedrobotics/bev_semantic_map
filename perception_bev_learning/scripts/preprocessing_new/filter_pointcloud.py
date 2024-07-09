import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def filter_point_cloud(cloud, y_threshold, x_threshold):
    # Create a list to hold the filtered points
    filtered_points = []

    # Iterate through the points in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        # point[1] is the y-coordinate
        # if point[1] <= y_threshold or (point[1] > y_threshold and point[0] < x_threshold):
        #     filtered_points.append(point)
        filtered_points.append(point)
    
    # Create a new PointCloud2 message with the filtered points
    filtered_cloud = pc2.create_cloud(cloud.header, cloud.fields, filtered_points)
    
    return filtered_cloud

def main(input_bag_path, output_bag_path, y_threshold, x_threshold):
    rospy.init_node('point_cloud_filter', anonymous=True)

    with rosbag.Bag(input_bag_path, 'r') as in_bag, rosbag.Bag(output_bag_path, 'w') as out_bag:
        for topic, msg, t in in_bag.read_messages():
            print(topic)
            if msg._type == 'sensor_msgs/PointCloud2':
                filtered_msg = filter_point_cloud(msg, y_threshold, x_threshold)
                out_bag.write(topic, filtered_msg, t)
            # else:
            #     out_bag.write(topic, msg, t)

if __name__ == "__main__":
    input_bag_path = '/home/manthan/bev_cover/halter_ranch_cover_v2_2024-06-02-02-03-38.bag'
    output_bag_path = '/home/manthan/bev_cover/gvom_v2.bag'
    y_threshold = 175  # Replace with your desired threshold value
    x_threshold = -152
    main(input_bag_path, output_bag_path, y_threshold, x_threshold)