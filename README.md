# eye_in_hand_calibration
EYE-IN-HAND calibration using opencv

# How to run
To execute, just run the "calibrate.py" file.

# Notes
Inside the images" folder, you will find all the different charuco photos.
Inside the "images_transformation_info.txt" file, you will find the corresponding poses of the gripper (gripper2base transformations).
Inside "freedrive.py", you can find a helper function that only works for a UR5e robotic arm, and prints the robot's gripper2base transformations, while enabling freedrive (moving the robot manually). It was used to gather the images for the calibration. 

All the images were taken in 1280*720 definition in my case.

# Motivation and rights
This is a streamlined solution of the problem of eye in hand calibration using opencv.

This was created due to the fact that no proper, clean documentation was found in the internet while I was searching for it.
Hopefully, it will accelerate the projects of many robotic engineers out there, who are dealing with robotic arm manipulation and need to perform a quick intrinsic and extrinsic calibration using a robot arm and an attached camera to the gripper.

Feel free to use the code in ANY way you want.
