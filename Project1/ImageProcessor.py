import time
from xmlrpc.client import MININT

import cv2 as cv
import numpy as np
import imutils
import os
from numpy.polynomial import Polynomial as poly

from cv2 import imshow


class ImageProcessor:

    def __init__(self):
        pass

    def undistort(self, folder, image, camera, destFolder=None):
        img = cv.imread(os.path.join(folder, image))
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


    def get_color_channels(self, img, masks=None):
        img_copy = img.copy()
        b, g, r = cv.split(img_copy)
        blank = np.zeros(img.shape[:2], dtype="uint8")


        #Gaussian Blur
        b_blur = cv.GaussianBlur(b, (3, 3), 0)
        g_blur = cv.GaussianBlur(g, (3, 3), 0)
        r_blur = cv.GaussianBlur(r, (3, 3), 0)

        b_edges = cv.Canny(b_blur, 25, 125, apertureSize=3)
        g_edges = cv.Canny(g_blur, 25, 125, apertureSize=3)
        r_edges = cv.Canny(r_blur, 25, 125, apertureSize=3)

        combthresh = cv.bitwise_or(b_edges, g_edges, r_edges)
        threshmask = cv.morphologyEx(masks[0], cv.MORPH_ERODE, (20, 20), iterations=2)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_DILATE, None, iterations=2)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_OPEN, (20, 20), iterations=2)

        invmask = cv.morphologyEx(masks[0], cv.MORPH_ERODE, None, iterations=10)
        invmask = cv.morphologyEx(invmask, cv.MORPH_OPEN, None, iterations=1)
        invmask = cv.bitwise_not(invmask)

        self.showimg(combthresh, "combthresh")
        self.showimg(threshmask, "threshmask")
        combthresh = cv.bitwise_and(combthresh, combthresh, mask=threshmask)
        combthresh = cv.bitwise_and(combthresh, combthresh, mask=invmask)
        self.showimg(combthresh, "edgesthresh")
        self.showimg(self.lines(combthresh, img_copy), "edgesthreshlines")

        return img_copy, b, g, r

    def lines(self, edge, img):

        #Approxpoly
        cons = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cons = imutils.grab_contours(cons)
        c = max(cons, key=cv.contourArea)
        #poly = cv.approxPolyDP(c, 0.02 * cv.arcLength(c, True), True)
        poly = cv.approxPolyN(c, 6, approxCurve=np.ndarray([]),ensure_convex=True)
        print(poly)
        cv.drawContours(img, [poly], 0, (0, 0, 255), 5)
        for index, point in enumerate(poly[0].astype(int)):
            cv.circle(img,(point[0], point[1]), 5, (0, 0, 255), -1)
            cv.putText(img, str(index), (point[0], point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

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

    def color_mask(self, img, lower, upper):
        img_copy = img.copy()
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_copy, lower, upper)
        mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)[1]
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None, iterations=2)
        self.showimg(mask, "mask")
        return mask

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


    def linesintersect(self, line1, line2):#
        intersections = list()
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        line1_params = poly.fit((x1, x2), (y1, y2), 1)
        line2_params = poly.fit((x3, x4), (y3, y4), 1)
        print(line1_params)
        print(line2_params)

    #Anzeigen des Bildes
    def showimg(self, img, name):
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyWindow(name)




