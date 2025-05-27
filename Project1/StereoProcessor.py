from numpy.f2py.auxfuncs import throw_error

import ImageProcessor
import StereoCamera
import cv2 as cv
import numpy as np


class StereoProcessor(ImageProcessor.ImageProcessor):

    def __init__(self):
        super().__init__()

    def undistort_rectify(self, frameL, frameR, stereocam):
        undistorted_left = cv.remap(frameL, stereocam.stereomapL_x, stereocam.stereomapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        undistorted_right = cv.remap(frameR, stereocam.stereomapR_x, stereocam.stereomapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        return undistorted_left, undistorted_right

    def find_depth(self, frame_left, frame_right, stereocam):

        #f von mm zu pixel umrechnen
        heightr, widthr, depthr = frame_right.shape
        heightl, widthl, depthl = frame_left.shape

        if widthl == widthr:
            f_pixel = (widthr * 0.5) / np.tan(stereocam.alpha * 0.5 * np.pi / 180)
        else:
            throw_error("width of left and right image do not match")

        xr = stereocam.right_camera.objPoints[0]
        xl = stereocam.left_camera.objPoints[0]

        disp = xl - xr
        zD = (stereocam.baseline*f_pixel) / disp

        return abs(zD)
