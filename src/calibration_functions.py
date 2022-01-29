import os
import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import re
import ast

SQUARE_LENGTH = 0.02562 # in m -> you have to count that in you printed charuco board
MARKER_LENGTH = 0.02056 # in m -> you have to count that in you printed charuco board
NUM_MARKERS = 17 # number of markers in charuco

def create_board():
    """
    Create a charuco board setting a specific seed for reproduceability
    """

    workdir = os.path.join(os.getcwd(), 'calibration')
    aruco_dict = aruco.Dictionary_create(nMarkers=17, markerSize=5, randomSeed=42)
    board = aruco.CharucoBoard_create(squaresX = 7, squaresY = 5, squareLength = SQUARE_LENGTH, markerLength = MARKER_LENGTH, dictionary = aruco_dict)
    imboard = board.draw((2000, 2000))
    cv2.imwrite("board.tiff", imboard)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    # ax.axis("off")
    # plt.show()

    return aruco_dict, board

def take_photos():
    """
    Load the images of the robot poses that you have already taken as a sorted list.
    """

    datadir = os.getcwd() + "/images/"
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
    order = np.argsort([int(p.split("/")[-1].split("_")[0]) for p in images])
    images = images[order]

    im = PIL.Image.open(images[0])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(im)
    plt.show()

    return images

def read_chessboards(aruco_dict, board, images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                # The function iterates to find the sub-pixel accurate location of corners or radial saddle points 
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape

    return allCorners,allIds,imsize

def calibrate_camera(allCorners, allIds, imsize, board):
    """
    Calibrates the camera using the charuco dected corners.
    """
    print("CAMERA CALIBRATION...")

    # initialize with known calibration values
    cameraMatrixInit = np.array([[ 930.327,   0., 629.822],
                                [  0.,  930.327,  358.317],
                                [  0.,   0.,   1.]])

    # if you don't have them, you can initialize it like this
    # cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
    #                              [    0., 1000., imsize[1]/2.],
    #                              [    0.,    0.,        1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

def check_calibration_results(images, mtx, dist):
    """
    Check calibration results on an image of the charuco from a robot pose.
    Basically, just read an image, undistort it, and plot it.
    """
    i=12 # select image id
    plt.figure()
    frame = cv2.imread(images[i])
    img_undist = cv2.undistort(frame,mtx,dist,None)
    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()

def add_local_axis(frame, aruco_dict, corners, mtx, dist, ids):
    """
    Add local axis to every charuco marker to see if your calibration was correct.
    """
    size_of_marker =  MARKER_LENGTH # marker length in meters
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)
    length_of_axis = 0.1
    imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    for i in range(len(tvecs)):
        imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
    plt.figure()
    plt.imshow(imaxis)
    plt.grid()
    plt.show()

    return rvecs, tvecs, imaxis

# perform eye in hand calibration 
def calibrate_hand_eye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
    """
    Performs eye in hand calibration.
    """
    R_g2b = np.array(R_gripper2base)
    t_g2b = np.array(t_gripper2base)
    R_t2c = np.array(R_target2cam)
    t_t2c = np.array(t_target2cam)

    # transform rotation vectors to rotation matrices and invert transformation
    R_g2b = []
    for idx, r in enumerate(R_gripper2base):
        r,_ = cv2.Rodrigues(np.array(r))
        R_g2b.append(r)

    R_t2c = []            
    for r in R_target2cam:
        r,_ = cv2.Rodrigues(np.array(r))
        R_t2c.append(r)

    # transform to numpy arrays and  perform eye in hand calibration
    t_g2b = np.array(t_g2b)
    t_t2c = np.array(t_t2c)
    t_g2b = t_g2b.reshape((len(t_g2b),3))
    t_t2c = t_t2c.reshape((len(t_t2c),3)).squeeze()
    print("\nPerforming hand-eye calibration...")
    mth = cv2.CALIB_HAND_EYE_TSAI
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=mth)

    return R_c2g, t_c2g
    
# invert a transformation from a to be to transformation from b to a
def invert_transformation(R_a2b, t_a2b):
    """
    Inverts the given rotation matrix and translation vector
    """
    dim = len(np.shape(R_a2b))
    if dim == 2:
        R_b2a = np.transpose(R_a2b).reshape((3,3))
        t_b2a = np.matmul(-np.transpose(R_a2b),t_a2b)
    else:
        R_b2a = []
        t_b2a = []            
        for idx, r in enumerate(R_a2b):
            r,_ = cv2.Rodrigues(np.array(r))
            r = np.transpose(r).reshape((3,3))
            t = np.matmul(-np.transpose(r), t_a2b[idx])
            R_b2a.append(r)
            t_b2a.append(t)

    return R_b2a, t_b2a


def read_image_transforms():
    """
    Take the transformation information about robot poses from a txt file.
    Reads and converts it into 2 lists of lists (rotation and translation).
    """

    R_list = []
    t_list = []

    os.chdir("..")
    image_dir = os.path.join(os.getcwd(), 'calibration', 'images_transformation_info.txt')
    with open(image_dir) as f:
        lines = f.readlines()
        n = -1
        for line in lines:
            if not re.match(r'^\s*$', line):    
                line = line.strip('\n')
                line = line.strip('\t')
                if line[0]=='[':
                    line = ast.literal_eval(line)
                    #line = [n.strip() for n in line]
                    n+=1
                    if n%2 == 0:
                        # rotation here
                        r = line
                        R_list.append(r)
                    else:
                        t = line
                        t_list.append(t)

    return R_list, t_list
