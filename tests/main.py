from picamera2 import Picamera2
import cv2 as cv
import os
import time
from libcamera import controls
import threading


#Raspberry PI camera controls
camera_controls_1 = None
camera_controls_2 = None

#Raspberry PI camera still image configuration
camera_config_1 = {"size":(1920, 1080), "format": "RGB888"}
camera_config_2 = {"size":(1920, 1080), "format": "RGB888"}

#File Paths to Images



if __name__ == "__main__":
    #Setting image configurations and camera controls
    

    #Show Videostream of both cameras with option to save frames
    while True:
        frame1 = picam1.capture_array()
        frame2 = picam2.capture_array()
        cv.imshow("Camera1", frame1)
        cv.imshow("Camera2", frame2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite(os.path.join(save_path_camera_1, f"stream_{int(time.time())}.jpg"), frame1)
            cv.imwrite(os.path.join(SAVE_, f"stream_{int(time.time())}.jpg"), frame2)

    picam1.stop()
    picam2.stop()
    cv.destroyAllWindows()


