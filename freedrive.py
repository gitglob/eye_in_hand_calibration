import rtde_control
import time

def main():
    """
    This is a helper function to run when taking photos and
    saving robot poses for charuco calibration.
    This was done using an realsense camera D415 and a UR5e robotic arm.
    """

    # setup connection to robot
    rtde_c = rtde_control.RTDEControlInterface("172.22.22.2")
    
    # set free drive
    rtde_c.teachMode()

    # set a 5sec interval
    INTERVAL = 5
    last = time.time()
    while True:
        # get the orientation and translation of b2g
        p = rtde_c.getForwardKinematics()
        t = p[0:3]
        R = p[3:6]
        print("Rotation:\n", R)
        print("Translation:\n", t)

        next = last + INTERVAL
        time.sleep(next - time.time())  # it's ok to sleep negative time
        last = next

if __name__ == "__main__":
    main()