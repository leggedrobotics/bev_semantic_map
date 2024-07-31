import rosbag
from sensor_msgs.msg import CameraInfo
import rospy
import os
import fnmatch

def find_bag_files(directory, keyword):
    # List to store all bag files
    bag_files = []

    # Walk through the directory
    for root, dir, files in os.walk(directory):
        for file in files:
            # Check if the file is a bag file and contains the keyword
            if fnmatch.fnmatch(file, '*{}*.bag'.format(keyword)):
                # Add the file to the list
                bag_files.append(os.path.join(root, file))

    return bag_files

def modify_camera_info(input_bag_file, output_bag_file, 
                       K_front, D_front, D_front_type, camera_info_front_topic, 
                       K_rear, D_rear, D_rear_type, camera_info_rear_topic):
    # Open the input bag file
    with rosbag.Bag(input_bag_file, 'r') as input_bag:
        # Open the output bag file
        with rosbag.Bag(output_bag_file, 'w') as output_bag:
            for topic, msg, t in input_bag.read_messages():
                # Check if the topic is the front camera info topic and the message is of type CameraInfo
                if topic == camera_info_front_topic:
                    # Modify the front camera info message
                    msg.K = K_front
                    msg.D = D_front
                    msg.distortion_model = D_front_type
                    print(f"Front image")
                # Check if the topic is the rear camera info topic and the message is of type CameraInfo
                elif topic == camera_info_rear_topic:
                    # Modify the rear camera info message
                    msg.K = K_rear
                    msg.D = D_rear
                    msg.distortion_model = D_rear_type
                # Write the modified or unmodified message to the new bag
                output_bag.write(topic, msg, t)

if __name__ == "__main__":
    # Example usage
    input_bag_directory = "/home/patelm/Data/nature_hiking/2024-06-07-seealpsee/run1/2024-06-07-11-59-26_anymal-d020"
    keyword = "jetson"
    camera_info_front_topic = '/hdr_camera_front/camera_info'
    camera_info_rear_topic = '/hdr_camera_rear/camera_info'

    bag_files = find_bag_files(input_bag_directory, keyword)

    K_front = [512.11257225, 0, 968.75886685,
             0, 502.77627934, 644.72345463,
             0, 0, 1]
    D_front = [0.13058332, 0.01104646, 0.01195079, -0.00302817]
    D_front_type = 'equidistant'

    K_rear = K_front # TODO: Change this to the rear camera matrix
    D_rear = D_front
    D_rear_type = D_front_type

    for input_bag_file in bag_files:
        # Modify the camera info in the bag file
        output_bag_file = input_bag_file.replace(keyword, keyword + "_modified")
        modify_camera_info(input_bag_file, output_bag_file, K_front, D_front, D_front_type, camera_info_front_topic, K_rear, D_rear, D_rear_type, camera_info_rear_topic)

    print("Finished modifying camera info in the bag file.")
