from numpy.f2py.auxfuncs import throw_error
import StereoCamera
import cv2 as cv
import numpy as np


class StereoProcessor():

    def undistort_rectify(self, frameL, frameR, stereocam):
        undistorted_left = cv.remap(frameL, stereocam.stereomapL_x, stereocam.stereomapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        undistorted_right = cv.remap(frameR, stereocam.stereomapR_x, stereocam.stereomapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        return undistorted_left, undistorted_right

    def find_depth(self, pr, pl, frame_left, frame_right, stereocam):

        #f von mm zu pixel umrechnen
        heightr, widthr, depthr = frame_right.shape
        heightl, widthl, depthl = frame_left.shape

        if widthl == widthr:
            f_pixel = (widthr * 0.5) / np.tan(stereocam.alpha * 0.5 * np.pi / 180)
        else:
            f_pixel = stereocam.left_camera.f

        xr = pr[0]
        xl = pl[0]

        disp = xl - xr
        zD = (stereocam.baseline*f_pixel) / disp
        print(f"Distanz Kamera-Punkt: {abs(zD)}")
        return abs(zD)
