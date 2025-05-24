from Project1.Camera import Camera
import cv2 as cv
import numpy as np
import threading

class StereoCamera(Camera, Camera):
    def __init__(self, left_camera, right_camera):
        super().__init__()
        self.left_camera = left_camera
        self.right_camera = right_camera


    def stereo_calibration_rectification(self, res, flags = 0):
        #Stereo-Kalibrierung mit zwei Kameras // Stereocalibration with 2 cameras
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, R, T, E, F = cv.stereoCalibrate(self.left_camera.objPoints, self.left_camera.imgPoints, self.right_camera.imgPoints, self.left_camera.cameraMatrix, self.left_camera.dist, self.right_camera.cameraMatrix, self.right_camera.dist, res, criteria=criteria_stereo, flags=flags)

        #Stereo Rektifizierung, Unverzerrung // Stereo rectification, undistortion


    def stereo_stream(self):
        pass
