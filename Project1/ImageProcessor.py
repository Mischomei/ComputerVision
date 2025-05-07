import time

import cv2 as cv
import numpy as np
import imutils
import os

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


    def find_edges(self, image): #Zwischenzeitliche Kantenerkennung
        img = image.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        #ret, thresh = cv.threshold(gray_blur, 100, 255, cv.THRESH_BINARY)
        edges = cv.Canny(gray_blur, 25, 150)

        cv.imshow("image", edges)
        cv.waitKey(0)
        cv.destroyWindow("image")
        return edges



    def find_contours(self, image): # Zwischenzeitliche Markererkennung
        img = image

        #Kantenerkennung // Edge Detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(gray_blur, 25, 125, apertureSize=3)
        print(edges)

        #Konturenerkennung // Contour Detection
        contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)


        imgcopy = img.copy()
        cv.drawContours(imgcopy, contours, -1, (0, 255, 0), 2)

        cv.imshow("imagecopy", imgcopy)
        cv.waitKey(0)
        cv.destroyWindow("imagecopy")

        return contours

    def find_corners(self, img, edges):
        img_copy = img.copy()

        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        #gray_blur = cv.GaussianBlur(gray, (5, 5), 0)

        corners = cv.cornerHarris(gray, 2, 3, 0.04)

        dst = cv.dilate(corners, None)
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        img_copy[dst > 0.01 * dst.max()] = [0, 0, 255]

        res = np.hstack((centroids, corners))
        res = np.uint8(res)
        img_copy[res[:, 1], res[:, 0]] = [0, 0, 255]
        img_copy[res[:, 3], res[:, 2]] = [0, 255, 0]


        cv.imshow("image", img_copy)

        cv.waitKey(0)
        cv.destroyWindow("image")
        return corners

    def get_color_channels(self, img):
        img_copy = img.copy()
        b, g, r = cv.split(img_copy)
        blank = np.zeros(img.shape[:2], dtype="uint8")

        #Finde edges
        b_edges = cv.Canny(cv.GaussianBlur(b, (3,3), 0), 25, 125, apertureSize=3)
        g_edges = cv.Canny(cv.GaussianBlur(g, (3, 3), 0), 25, 125, apertureSize=3)
        r_edges = cv.Canny(cv.GaussianBlur(r, (3, 3), 0), 25, 125, apertureSize=3)
        cv.imshow("blue", b_edges)
        cv.imshow("green", g_edges)
        cv.imshow("red", r_edges)

        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow("image", cv.bitwise_or(b_edges, g_edges, r_edges))
        cv.waitKey(0)
        cv.destroyWindow("image")

        return b, g, r

    def calc_coords_from_edges(self):
        pass

    def calc_coords_from_corners(self):
        pass




