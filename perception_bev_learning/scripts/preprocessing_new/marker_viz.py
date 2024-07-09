import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import tf


def create_frustum_marker(camera_intrinsics, camera_extrinsics, marker_id=0, d=0.3):
    fx, fy, cx, cy = camera_intrinsics
    width, height = 2 * cx, 2 * cy

    # Define image plane corners in camera frame
    corners = np.array([
        [-cx / fx, -cy/fy, 1],
        [(width - cx)/fx , -cy/fy, 1],
        [(-cx/ fx), (height - cy)/fy, 1],
        [(width - cx )/fx, (height - cy)/fy, 1]
    ])
    corners = corners * d

    # Convert corners to homogeneous coordinates
    corners_hom = np.hstack((corners, np.ones((corners.shape[0], 1))))

    # Apply camera extrinsics to transform to world frame
    corners_world = (camera_extrinsics @ corners_hom.T).T[:, :3]

    # Create the marker
    marker = Marker()
    marker.header.frame_id = "crl_rzr/map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "frustum"
    marker.id = marker_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.5
    # marker.pose.position.x = camera_extrinsics[0,3]
    # marker.pose.position.y = camera_extrinsics[1,3]
    # marker.pose.position.z = camera_extrinsics[2,3]
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1

    # Camera origin
    points = []
    camera_origin = Point(camera_extrinsics[0,3], camera_extrinsics[1,3], camera_extrinsics[2,3])
    # print(camera_extrinsics)
    # camera_origin = Point(0,0,0)

    # # Add lines from camera origin to frustum corners
    # for corner in corners_world:
    #     p = Point(corner[0], corner[1], corner[2])
    #     marker.points.append(camera_origin)
    #     marker.points.append(p)

    # # Add lines between frustum corners
    # for i in range(len(corners_world)):
    #     for j in range(i + 1, len(corners_world)):
    #         p1 = Point(corners_world[i][0], corners_world[i][1], corners_world[i][2])
    #         p2 = Point(corners_world[j][0], corners_world[j][1], corners_world[j][2])
    #         marker.points.append(p1)
    #         marker.points.append(p2)
    for i in range(4):
        next_i = (i + 1) % 4
        points.extend([camera_origin,
                       Point(corners_world[i][0], corners_world[i][1], corners_world[i][2]),
                       Point(corners_world[next_i][0], corners_world[next_i][1], corners_world[next_i][2])])
    
    # Add triangles for the image plane
    points.extend([
        Point(corners_world[0][0], corners_world[0][1], corners_world[0][2]),
        Point(corners_world[1][0], corners_world[1][1], corners_world[1][2]),
        Point(corners_world[2][0], corners_world[2][1], corners_world[2][2]),
        
        Point(corners_world[1][0], corners_world[1][1], corners_world[1][2]),
        Point(corners_world[2][0], corners_world[2][1], corners_world[2][2]),
        Point(corners_world[3][0], corners_world[3][1], corners_world[3][2])
    ])

    # Add points to the marker
    marker.points = points


    return marker

def get_camera_extrinsics(tf_listener, source_frame, target_frame):
    try:
        tf_listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(4.0))
        (trans, rot) = tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
        # Create transformation matrix
        trans_matrix = tf.transformations.translation_matrix(trans)
        rot_matrix = tf.transformations.quaternion_matrix(rot)
        extrinsics = np.dot(trans_matrix, rot_matrix)
        return extrinsics
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr("Could not get transform from %s to %s" % (source_frame, target_frame))
        return None


def main():
    print("Init Node")
    rospy.init_node('camera_frustum_node')

    marker_pub = []
    for i in range(4):
        marker_pub.append(rospy.Publisher(f'viz_marker_{i}', Marker, queue_size=1))

    # Camera intrinsics (example values)
    fx, fy = 419.45 , 419.45
    cx, cy = 480, 300
    camera_intrinsics = (fx, fy, cx, cy)

    # Create TF listener
    tf_listener = tf.TransformListener()

    # Get camera extrinsics from map to camera frame
    source_frame = "crl_rzr/map"
    target_frames = ["crl_rzr/multisense_front/aux_camera_frame", "crl_rzr/multisense_back/aux_camera_frame",
                      "crl_rzr/multisense_left/aux_camera_frame", "crl_rzr/multisense_right/aux_camera_frame"]

    rate = rospy.Rate(10)
    print("Starting loop")
    while not rospy.is_shutdown():
        for i in range(4):
            camera_extrinsics = get_camera_extrinsics(tf_listener, source_frame, target_frames[i])
            if camera_extrinsics is None:
                rospy.logerr("Failed to get camera extrinsics. Exiting...")
                return
            # Create the frustum marker
            marker = create_frustum_marker(camera_intrinsics, camera_extrinsics)
            marker_pub[i].publish(marker)
        rate.sleep()

if __name__ == "__main__":
    main()
