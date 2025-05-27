from Project1.Camera import Camera
import cv2 as cv
import numpy as np
import threading

class StereoCamera(Camera):
    stereomapL_x, stereomapL_y = None, None
    stereomapR_x, stereomapR_y = None, None
    baseline = None
    alpha = None
    def __init__(self, left_camera, right_camera, baseline, alpha):
        super().__init__()
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.baseline = baseline
        self.alpha = alpha


    def stereo_calibration_rectification(self, res, flags = 0):
        #Stereo-Kalibrierung mit zwei Kameras // Stereocalibration with 2 cameras
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, R, T, E, F = cv.stereoCalibrate(self.left_camera.objPoints, self.left_camera.imgPoints, self.right_camera.imgPoints, self.left_camera.newCameramatrix, self.left_camera.dist, self.right_camera.newCameramatrix, self.right_camera.dist, res, criteria=criteria_stereo, flags=flags)

        rectscale = 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, res, R, T, rectscale, (0,0))

        stereomapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, res, cv.CV_16SC2)
        stereomapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, res, cv.CV_16SC2)

        if stereomapR and stereomapL:
            self.stereomapL_x = stereomapL[0]
            self.stereomapL_y = stereomapL[1]
            self.stereomapR_x = stereomapR[0]
            self.stereomapR_y = stereomapR[1]
            self.stereomapR = stereomapR
            print("Stereo-Kalibrierung erfolgreich")
        else:
            print("Stereo-Kalibrierung fehlgeschlagen")

        #Speicherung der ganzen Parameter
        cv_file = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_WRITE)
        cv_file.write("stereoMapL_x", stereomapL[0])
        cv_file.write("stereoMapL_y", stereomapL[1])
        cv_file.write("stereoMapR_x", stereomapR[0])
        cv_file.write("stereoMapR_y", stereomapR[1])
        cv_file.release()

    def read_map(self):
        pass

    def stereo_stream(self):
        pass
