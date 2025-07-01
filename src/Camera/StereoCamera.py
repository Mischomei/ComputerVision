import os

from src.Camera.Camera import Camera
import cv2 as cv


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
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, res, R, T, rectscale, (0,0), alpha=0)

        stereomapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, res, cv.CV_16SC2)
        stereomapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, res, cv.CV_16SC2)

        if stereomapR and stereomapL:
            self.stereomapL_x = stereomapL[0]
            self.stereomapL_y = stereomapL[1]
            self.stereomapR_x = stereomapR[0]
            self.stereomapR_y = stereomapR[1]
            print("Stereo-Kalibrierung erfolgreich")
        else:
            print("Stereo-Kalibrierung fehlgeschlagen")


    #TODO Fix Settings Saving and Loading
    def save_map(self, dic=""):
        # Speicherung der ganzen Parameter
        cv_file = cv.FileStorage(os.path.join(dic,"stereoMap.xml"), cv.FILE_STORAGE_WRITE)
        cv_file.write("stereoMapL_x", self.stereomapL_x)
        cv_file.write("stereoMapL_y", self.stereomapL_y)
        cv_file.write("stereoMapR_x", self.stereomapR_x)
        cv_file.write("stereoMapR_y", self.stereomapR_y)
        cv_file.release()

    def read_map(self, dic=""):
        cv_file = cv.FileStorage()
        cv_file.open(os.path.join(dic,"stereoMap.xml"), cv.FILE_STORAGE_READ)

        self.stereomapL_x = cv_file.getNode("stereoMapL_x").mat()
        self.stereomapL_y = cv_file.getNode("stereoMapL_y").mat()
        self.stereomapR_x = cv_file.getNode("stereoMapR_x").mat()
        self.stereomapR_y = cv_file.getNode("stereoMapR_y").mat()
        cv_file.release()


    def stereo_stream(self):
        pass
