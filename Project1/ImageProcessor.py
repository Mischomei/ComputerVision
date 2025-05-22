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
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(camera.cameraMatrix, camera.dist, (w,h), 1, (w, h))
        #Unverzerrung
        dest = cv.undistort(img, camera.cameraMatrix, camera.dist, None, newCameraMatrix)
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


    def find_edges(self, image, masks): #Zwischenzeitliche Kantenerkennung
        img = image.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        #ret, thresh = cv.threshold(gray_blur, 100, 255, cv.THRESH_BINARY)
        edges = cv.Canny(gray_blur, 25, 150)
        if masks:
            for mask in masks:
                copycopy = img.copy()

                comb = cv.bitwise_and(edges, edges, mask=mask)
                self.showimg(self.lines(comb, img), "edges")
        return edges

    def get_color_channels(self, img, masks=None):
        img_copy = img.copy()
        b, g, r = cv.split(img_copy)
        blank = np.zeros(img.shape[:2], dtype="uint8")

        #Gaussian Blur
        b_blur = cv.GaussianBlur(b, (5, 5), 0)
        g_blur = cv.GaussianBlur(g, (5, 5), 0)
        r_blur = cv.GaussianBlur(r, (5, 5), 0)

        #Thresholding und Maske für Kantenerkennung
        b_thresh = cv.threshold(b_blur, 95, 255, cv.THRESH_BINARY_INV)[1]
        g_thresh = cv.threshold(g_blur, 95, 255, cv.THRESH_BINARY_INV)[1]
        r_thresh = cv.threshold(r_blur, 95, 255, cv.THRESH_BINARY_INV)[1]
        combthresh = cv.bitwise_or(b_thresh, g_thresh, r_thresh)
        threshmask = cv.morphologyEx(masks[0], cv.MORPH_ERODE, (20, 20), iterations=2)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_CLOSE, None, iterations=10)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_OPEN, (20, 20), iterations=1)
        self.showimg(threshmask, "threshmask")
        combthresh = cv.bitwise_and(combthresh, combthresh, mask=threshmask)
        edgesthresh = cv.Canny(combthresh, 25, 150, apertureSize=3)
        self.showimg(edgesthresh, "edgesthresh")
        self.showimg(self.lines(edgesthresh, img_copy), "edgesthreshlines")

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
        self.showimg(img, "lines")

        for index, point in enumerate(poly[0].astype(int)):
            cv.circle(img,(point[0], point[1]), 5, (0, 0, 255), -1)
            cv.putText(img, str(index), (point[0], point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.showimg(img, "lines")
        #Hough Line Transform // Hough-Line-Transformation
        imgcopy = img.copy()
        edges = edge.copy()
        rho = 1  # Abstandauflösung in Pixel
        theta = np.pi / 180  # Winkelauflösung in Radiant
        threshold = 20  # Mindestanzahl an Stimmen (Schnittpunkte in der Hough-Ebene)
        min_line_length = 30  # Minimale Linienlänge. Linien kürzer als diese werden verworfen.
        max_line_gap = 10 # Maximaler Abstand zwischen Liniensegmenten, um als eine Linie betrachtet zu werden.

        lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Linien auf ein leeres Bild oder das Originalbild zeichnen (optional)
        line_image = np.copy(imgcopy) * 0  # Erzeuge ein schwarzes Bild
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(imgcopy, (x1, y1), (x2, y2), (255, 0, 0), 3)

        #Corners from Lines
        px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, px6, py6, px7, py7, px8, py8 = line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1], line[0][0], line[0][1]


        for line in lines:
            x1, y1, x2, y2 = line[0]
            if px1 > x1:
                px1 = x1
                py1 = y1
                px8 = x2
                py8 = y2
            if px2 < x2:
                px2 = x2
                py2 = y2
                px7 = x1
                py7 = y1
            if py3 > y1:
                py3 = y1
                px3 = x1
                px6 = x2
                py6 = y2
            if py4 < y2:
                py4 = y2
                px4 = x2
                px5 = x1
                py5 = y1


        cv.circle(imgcopy, (px1, py1), 5, (0, 0, 0), -1)
        #cv.circle(imgcopy, (px2, py2), 5, (0, 0, 0), -1)
        cv.circle(imgcopy, (px3, py3), 5, (0, 0, 0), -1)
        #cv.circle(imgcopy, (px4, py4), 5, (0, 0, 0), -1)
        #cv.circle(imgcopy, (px5, py5), 5, (0, 0, 255), -1)
        cv.circle(imgcopy, (px6, py6), 5, (0, 255, 0), -1)
        #cv.circle(imgcopy, (px7, py7), 5, (255, 255, 255), -1)
        cv.circle(imgcopy, (px8, py8), 5, (0, 255, 255), -1)


        return imgcopy

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

    def color_mask(self, img, lower, upper, preprocess=False):
        img_copy = img.copy()
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_copy, lower, upper)
        mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)[1]

        self.showimg(mask, "mask")

        return mask

    def cornersto3d(self, corners):
        if len(corners) == 4:
            pass

        #AUs 6 erkennbaren des Quaders ein Sechseck und daraus seine 3D koordinaten
        if len(corners) == 6:
            # Aus Ecken Kanten machen
            pass


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




