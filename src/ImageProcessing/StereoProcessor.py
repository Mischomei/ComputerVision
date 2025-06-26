import cv2 as cv
import numpy as np
from numpy.f2py.auxfuncs import throw_error

from src.ImageProcessing.ImageProcessor import ImageProcessor


class StereoProcessor(ImageProcessor):
    debug = False
    def __init__(self, debug=False):
        super().__init__(debug)
        self.debug = debug

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

        xr = pr
        xl = pl

        disp = pr - pl
        zD = (stereocam.baseline*f_pixel) / disp[0]
        print(f"Distanz Kamera-Punkt: {abs(zD)}")
        return abs(zD)

    def cons_stereo(self, imgleft, imgright, masks):
        imgcopyr = imgright.copy()
        imgcopyl = imgleft.copy()
        leftmasks = list()
        rightmasks = list()
        for mask in masks:
            leftmasks.append((mask[0], mask[2]))
            rightmasks.append((mask[1], mask[2]))

        rightimg = self.cons_per_color(imgcopyr, rightmasks)
        if self.debug: self.showimg(rightimg, "cons_stereo_right")
        leftimg = self.cons_per_color(imgcopyl, leftmasks)
        if self.debug: self.showimg(leftimg, "cons_stereo_left")
        return rightimg, leftimg


    def combine_masks(self, leftmasks, rightmasks):
        outmasks = list()

        if leftmasks.keys() == rightmasks.keys():
            for key in leftmasks.keys():
                outmasks.append((leftmasks[key], rightmasks[key], key))
        else:
            throw_error("Rechte und Linke Maskenlisten haben nicht die gleichen Farben")

        return outmasks

