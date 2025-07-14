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

        #if widthl == widthr:
            #f_pixel = (widthr * 0.5) / np.tan(stereocam.alpha * 0.5 * np.pi / 180)
        #else:
        f_pixel = stereocam.left_camera.cameraMatrix[0, 0]

        xr = pr
        xl = pl

        disp = pr - pl
        zD = (stereocam.baseline*f_pixel) / disp[0]
        print(f"Distanz Kamera-Punkt: {abs(zD)}")
        return abs(zD)

    def calcdist(self, ap1, ap2, p1, p2):
        apdist = 0.4
        vec1 = ap2 - ap1
        vec2 = p1 - ap1

        deg = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        rel = np.linalg.norm(abs(ap1- ap2), ord=2) / apdist

        distpap = np.linalg.norm(abs(p1 - ap1), ord=2) * rel
        x = distpap * np.cos(deg)
        y = distpap * np.sin(deg)


        return (x, y)




    def cons_stereo(self, imgleft, imgright, masks):
        imgcopyr = imgright.copy()
        imgcopyl = imgleft.copy()
        leftmasks = list()
        rightmasks = list()
        for mask in masks:
            leftmasks.append((mask[0], mask[2]))
            rightmasks.append((mask[1], mask[2]))

        rightimg, rightpoints = self.cons_per_color(imgcopyr, rightmasks)
        if self.debug: self.showimg(rightimg, "cons_stereo_right")
        leftimg, leftpoints = self.cons_per_color(imgcopyl, leftmasks)
        if self.debug: self.showimg(leftimg, "cons_stereo_left")
        return rightimg, leftimg, rightpoints, leftpoints


    def combine_masks(self, leftmasks, rightmasks):
        outmasks = list()

        if leftmasks.keys() == rightmasks.keys():
            for key in leftmasks.keys():
                outmasks.append((leftmasks[key], rightmasks[key], key))
        else:
            throw_error("Rechte und Linke Maskenlisten haben nicht die gleichen Farben")

        return outmasks

