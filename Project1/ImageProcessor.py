import cv2 as cv
import numpy as np
import imutils
import os

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


    def find_marker(self, image):
        img = image

        #Kantenerkennung // Edge Detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(gray_blur, 50, 150)

        contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv.contourArea)

        return cv.minAreaRect(c)

    def draw_marker(self, marker, image):
        box = cv.BoxPoints(marker) if imutils.is_cv2() else cv.boxPoints(marker)
        box = np.int32(box)
        cv.drawContours(image, [box], -1, (0, 255, 0), 2)

        cv.imshow("image", image)
        cv.waitKey(0)