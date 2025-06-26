import time
from xmlrpc.client import MININT

import cv2 as cv
import numpy as np
import imutils
import os
from pathlib import Path
from numpy.polynomial import Polynomial as poly

from cv2 import imshow, invert

MASK_REP = 5
CAMERA_PARAM_REP = 5

class ImageProcessor:
    debug = False

    def __init__(self, debug=False):
        self.debug = debug


    def undistort(self, folder, image, camera, destFolder=None):
        img = cv.imread(folder / image)
        #debug
        if self.debug:self.showimg(img, "img")
        h,w = img.shape[:2]

        #Optimisierung der Kameramatrix passend zur Auflösung des Bildes
        newcameramatrix, roi = cv.getOptimalNewCameraMatrix(camera.newCameramatrix, camera.dist, (w,h), 1, (w, h))
        #Unverzerrung
        dest = cv.undistort(img, camera.newCameramatrix, camera.dist, None, newcameramatrix)
        x, y, w, h = roi
        dest = dest[y:y+h, x:x+w]
        if destFolder:
            cv.imwrite(os.path.join(destFolder, f"undistorted_{image}"), dest)

        return dest


    def reprojectionerror(self, camera, objPoints, imgPoints):
        mean_error = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv.projectPoints(objPoints[i], camera.rvecs[i], camera.tvecs[i], camera.cameraMatrix, camera.dist)
            error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints)
            mean_error += error
        print(f"Fehler: {mean_error/len(objPoints)}")


    def get_color_channels(self, img, mask):
        distparam = 3
        img_copy = img.copy()
        b, g, r = cv.split(img_copy)
        blank = np.zeros(img.shape[:2], dtype="uint8")


        #Gaussian Blur
        b_blur = cv.GaussianBlur(b, (3, 3), 0)
        g_blur = cv.GaussianBlur(g, (3, 3), 0)
        r_blur = cv.GaussianBlur(r, (3, 3), 0)

        #Edge detection on colorchannels
        b_edges = cv.Canny(b_blur, 25, 125, apertureSize=3)
        g_edges = cv.Canny(g_blur, 25, 125, apertureSize=3)
        r_edges = cv.Canny(r_blur, 25, 125, apertureSize=3)


        combthresh = cv.bitwise_or(b_edges, g_edges, r_edges)
        combthresh = cv.morphologyEx(combthresh, cv.MORPH_CLOSE, (5, 5), iterations=3)
        threshmask = self.edit_thresh_mask(mask, distparam)
        invmask = self.edit_inv_mask(mask, distparam)


        combthresh = cv.bitwise_and(combthresh, combthresh, mask=threshmask)
        combthresh = cv.bitwise_and(combthresh, combthresh, mask=invmask)

        #debug
        if self.debug: self.showimg(combthresh, "combthresh")

        return combthresh


    def lines(self, edge):

        #Approxpoly
        cons = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cons = imutils.grab_contours(cons)
        c = max(cons, key=cv.contourArea)

        #poly = cv.approxPolyDP(c, 0.02 * cv.arcLength(c, True), True)
        poly = cv.approxPolyN(c, 6, approxCurve=np.ndarray([]),ensure_convex=True)

        #debug
        if self.debug: print(poly)

        return poly, np.asarray(poly[0].astype(int))


    def drawcontour(self, img, points, poly=None, color=None):
        copyimg = img.copy()

        if poly is not None:
            cv.drawContours(copyimg, [poly], 0, (255, 0, 0), 5)
        else:
            for index, point in enumerate(points):
                if index+1 < np.size(points, 0):
                    cv.line(copyimg, points[index], points[index+1], (255, 0, 0), 5)
                else:
                    cv.line(copyimg, points[index], points[0], (255, 0, 0), 5)
        for index, point in enumerate(points):
            cv.circle(copyimg, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv.putText(copyimg, str(index), (point[0] + 5, point[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if color:
            minpoint = min(points, key = lambda t:t[1])
            cv.putText(copyimg, color, minpoint-(15, 15), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

        return copyimg


    #Drehen vom Bild
    def rotate(self, img, angle=90):
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return img


    #Croppen vom Bild
    def crop(self, img, cropx_s=None, cropx_e=None, cropy_s=None, cropy_e=None):
        cxs = cropx_s if cropx_s else 0
        cxe = cropx_e if cropx_e else img.shape[1]
        cys = cropy_s if cropy_s else 0
        cye = cropy_e if cropy_e else img.shape[0]
        img_copy = img.copy()
        return img_copy[cxs:cxe, cys:cye]


    def create_mask(self, img, lower, upper):
        img_copy = img.copy()
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_copy, lower, upper)
        mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)[1]
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None, iterations=2)
        if self.debug: self.showimg(mask, "created mask")

        return mask



    def edit_thresh_mask(self, mask, dist_param):
        #Dilating mask to create outer object selection
        threshmask = cv.morphologyEx(mask, cv.MORPH_ERODE, (20, 20), iterations=2)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_DILATE, None, iterations=dist_param)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_OPEN, (20, 20), iterations=2)

        #debug
        if self.debug: self.showimg(threshmask, "threshmask")
        return threshmask


    def edit_inv_mask(self, mask, dist_param):
        #Eroding Mask to create inner removal mask
        invmask = cv.morphologyEx(masks[0], cv.MORPH_ERODE, None, iterations=dist_param*2)
        invmask = cv.morphologyEx(invmask, cv.MORPH_OPEN, None, iterations=1)
        invmask = cv.bitwise_not(invmask)

        #debug
        if self.debug: self.showimg(invmask, "invmask")
        return invmask


    def edit_inv_mask(self, mask, dist_param):
        pass


    def cornersto3d(self, corners):
        if len(corners) == 4:
            pass

        #AUs 6 erkennbaren des Quaders ein Sechseck und daraus seine 3D koordinaten
        if len(corners) == 6:
            # Aus Ecken Kanten machen
            pass


    def depthestimatetest(self, img, realsize, imgsize, f, sensor):
        return ()


    def refine_edges(self):
        pass


    def calc_coords_from_edges(self):
        pass


    def calc_coords_from_corners(self):
        pass


    def detect_arucomarker(self, img):
        pass


    def createmasks(self, img, colors):
        imgcopy = img.copy()
        outmasks = dict()

        for color in colors:
            mask = self.create_mask(imgcopy, color[0], color[1])
            if cv.hasNonZero(mask):
                outmasks[color[2]] = mask

        return outmasks

    #Für alle Farben jeweils seperate Kantenerkennungen und diese kombinieren
    def cons_per_color(self, img, masks):
        copyimg = img.copy()
        outimg = img.copy()

        for mask in masks:
            edges = self.get_color_channels(copyimg, mask[0])
            polyn, points = self.lines(edges)
            outimg = self.drawcontour(outimg, points, polyn, mask[1])

        #debug
        if self.debug: self.showimg(outimg, "cons_per_color")

        return outimg


    #Anzeigen des Bildes
    @staticmethod
    def showimg(img, name):
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyWindow(name)


    def sbm(self):
        out = np.array()
        while out.empty():
            for i in range(MASK_REP):
                try:
                    pass
                except:
                    pass



    def start_searching(self):
        pass




