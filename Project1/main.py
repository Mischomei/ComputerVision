import Camera
import ImageProcessor
import StereoCamera
from picamera2 import Picamera2
import cv2 as cv
import os
import time
from libcamera import controls

#Raspberry PI camera controls
camera_controls_1 = {}
camera_controls_2 = {}

#Raspberry PI camera still image configuration
camera_config_1 = {"size":(1920, 1080), "format": "BGR888"}
camera_config_2 = {"size":(1920, 1080), "format": "BGR888"}

#File Paths to Images
calibration_camera_1 = "example_data/calibration_left"
calibration_camera_2 = "example_data/calibration_right"
container_images_camera_1 = "example_data/container_left"
container_images_camera_2 = "example_data/container_right"
save_path_camera_1 = "example_data/calibration_left"
save_path_camera_2 = "example_data/calibration_right"


if __name__ == "__main__":
    #Setting image configurations and camera controls
    picam1 = Picamera2(0)
    picam2 = Picamera2(1)
    picam1.configure(camera_config_1)
    picam2.configure(camera_config_2)
    picam1.set_controls(camera_controls_1)
    picam2.set_controls(camera_controls_2)
    picam1.start()
    picam2.start()

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
            cv.imwrite(os.path.join(save_path_camera_2, f"stream_{int(time.time())}.jpg"), frame2)

    picam1.stop()
    picam2.stop()
    cv.destroyAllWindows()


