import numpy as np
import cv2, PIL, os
from src.calibration_functions import *

def main():

    #Camera pose estimation using CHARUCO chessboard

    aruco_dict, board = create_board()
    #aruco_dict = read_board() # doesn't exist wtf

    images = take_photos()
    
    allCorners, allIds, imsize = read_chessboards(aruco_dict, board, images)

    # this is correct according to https://docs.opencv.org/4.x/d9/d6a/group__aruco.html
    ret, mtx, dist, R_t2c, t_t2c = calibrate_camera(allCorners, allIds, imsize, board)
    R_t2c = np.array(R_t2c).squeeze()
    t_t2c = np.array(t_t2c).squeeze()
    tool_poses = np.concatenate([R_t2c,t_t2c], 1)
    tool_poses_df = pd.DataFrame(data=tool_poses, columns=["rx", "ry", "rz","tx", "ty", "tz"])
    tool_poses_df.index.name = "image"
    print("\nTarget2camera transformation:")
    print("Returned value:\n", ret)
    print("Distortion coefficients:\n", dist)
    print("Intrinsics matrix:\n", mtx)
    print("Transformations:\n", tool_poses_df)

    check_calibration_results(images, mtx, dist)

    frame = read_undistort_pic()
    corners, ids, frame_markers = post_process(aruco_dict, frame)
    # # Use of camera calibration to estimate 3D translation and rotation of each marker on a scene
    #R_t2c, t_t2c, imaxis = add_local_axis(frame, aruco_dict, corners, mtx, dist, ids)
    #convert_to_df(R_t2c, t_t2c, ids, imaxis, corners)
    #print_pixels(corners, ids)

    # The following lines provide an alternative approach that can be used, were we consider each marker coordinate frame as the target frame
    # for the eye-in hand calibration, therefore we have num_images*num_markers transformation matrices
    # then, you can just take the average values of the transformation matrices for each marker
    # img_dict, marker_dict = create_dictionaries(images, mtx, dist, aruco_dict, corners, ids, frame_markers)
    # R_cam2tool_dict, mean_R_cam2tool, t_cam2tool_dict, mean_t_cam2tool = per_marker(marker_dict, tool_poses_df)
    # R_cam2tool, t_cam2tool = all_markers(tool_poses_df, marker_dict)
    # compare_methods(R_cam2tool, t_cam2tool, mean_R_cam2tool, mean_t_cam2tool)
    #print((mean_R_cam2tool-R_cam2tool)*100/R_cam2tool,(mean_t_cam2tool-t_cam2tool)*100/t_cam2tool)
    #print(mean_t_cam2tool)
    #print(t_cam2tool)

    # Read pose transforms from "images_transformation_info.txt"
    R_g2b, t_g2b = read_image_transforms()
    
    # perform eye_in_hand calibration to calculate cam2gripper transform
    R_c2g, t_c2g = calibrate_hand_eye(R_gripper2base=R_g2b, t_gripper2base=t_g2b, R_target2cam=R_t2c, t_target2cam=t_t2c)
    print("Cam2gripper transformation...")
    print("\t\tRotation matrix:\n", R_c2g)
    print("\t\tTranslation vector:\n", t_c2g)


if __name__ == "__main__":
    main()