import numpy as np
import cv2, PIL, os
from src.calibration_functions import *

def main():

    #Camera pose estimation using CHARUCO chessboard
    aruco_dict, board = create_board()

    images = take_photos()
    
    allCorners, allIds, imsize = read_chessboards(aruco_dict, board, images)

    # this is correct according to https://docs.opencv.org/4.x/d9/d6a/group__aruco.html
    ret, mtx, dist, R_t2c, t_t2c = calibrate_camera(allCorners, allIds, imsize, board)
    R_t2c = np.array(R_t2c).squeeze()
    t_t2c = np.array(t_t2c).squeeze()
    print("\nTarget2camera transformation:")
    print("Returned value:\n", ret)
    print("Distortion coefficients:\n", dist)
    print("Intrinsics matrix:\n", mtx)

    check_calibration_results(images, mtx, dist)

    frame = read_undistort_pic()
    corners, ids, frame_markers = post_process(aruco_dict, frame)
    # # Use of camera calibration to estimate 3D translation and rotation of each marker on a scene (the scene described by the image "frame")
    # rvecs, tvecs, imaxis = add_local_axis(frame, aruco_dict, corners, mtx, dist, ids)

    # Read pose transforms from "images_transformation_info.txt"
    R_g2b, t_g2b = read_image_transforms()
    
    # perform eye_in_hand calibration to calculate cam2gripper transform
    R_c2g, t_c2g = calibrate_hand_eye(R_gripper2base=R_g2b, t_gripper2base=t_g2b, R_target2cam=R_t2c, t_target2cam=t_t2c)
    print("Cam2gripper transformation...")
    print("\t\tRotation matrix:\n", R_c2g)
    print("\t\tTranslation vector:\n", t_c2g)

    # # calculate target2base transform to confirm
    print("\n\nTarget2base transformation (default position):")
    R_t2c_0,_ = cv2.Rodrigues(np.array(R_t2c[0]))
    R_g2b_0,_ = cv2.Rodrigues(np.array(R_g2b[0]))
    R_t2b = R_t2c_0 * R_c2g * R_g2b_0
    t_t2b = np.array(t_t2c[0]).reshape((3,1)) + np.array(t_c2g).reshape((3,1)) + np.array(t_g2b[0]).reshape((3,1))
    print("Rotation matrix:\n", R_t2b)
    print("Translation vector:\n", t_t2b)

if __name__ == "__main__":
    main()