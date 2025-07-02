import glob
import cv2 as cv
import numpy as np
import os
import json
import time
from pathlib import Path

# Kamera Klasse
class Camera:
    debug=False
    #Kameraparameter // Camera parameters
    f = None # Focal length
    resolution = (0, 0) # Auflösung // Resolution
    cameraMatrix, dist, rvecs, tvecs = None, None, None ,None # Kameramatrix, Verzerrungsmatrix, Rotationsvektoren, Verschiebungsvektoren // Camera Matrix, Distortion Matrix, Rotation Vectors, Translation Vectors
    objPoints, imgPoints = None, None
    newCameramatrix = None

    def __init__(self, data=None, debug=False):
        self.debug = debug
        if data:
            self.load_settings(data)


    def calibrate(self, folder, res, checkboardSize, calibfilesdtype="jpg"):
        # Kriterien für Terminierung des Kalibrierungvorgangs // termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Weltkoordinaten für spätere Verwendung mit numpy (???) // ObjectPoints for later use
        objp = np.zeros((checkboardSize[0] * checkboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkboardSize[0], 0:checkboardSize[1]].T.reshape(-1, 2)

        # Arrays für Weltkoordinaten (3D Punkte in Reallife) und Bildkoordinaten (2D-Punkte auf Bildebene) // Arrays for Object points and Image points
        objPoints = []
        imgPoints = []

        # Bilder für Kalibrierung // Calibration images

        images = list(folder.glob(f"*.{calibfilesdtype}"))
        for image in images:
            print(image)
            img = cv.imread(image)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Erkennen der Ecken auf Schachbrettmuster // Corner detection on checkboard
            ret, corners = cv.findChessboardCorners(img_gray, checkboardSize, None)

            if ret:
                objPoints.append(objp)

                # Genauere Eckenerkennung im Subpixelbereich // Subpixel corner detection
                corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                imgPoints.append(corners2)

                cv.drawChessboardCorners(img, checkboardSize, corners2, ret)

        cv.destroyAllWindows()

        # Kalibrierung // Calibration
        height, width, channels = img.shape
        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, (width, height), None, None)

        newCameramatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
        newCameramatrix = cameraMatrix
        if ret:
            self.cameraMatrix = cameraMatrix
            self.newCameramatrix = newCameramatrix
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.f = cameraMatrix[0, 0]
            self.imgPoints = imgPoints
            self.objPoints = objPoints
            print(f"---Kamera kalibriert: {ret}---")
        else:
            print(f"---Kalibrierung fehlgeschlagen---")

        return ret


    def show_parameters(self):
        print(f"---Focal Length: {self.f}---")
        print(f"---Camera Matrix:---\n {self.cameraMatrix}")
        print(f"---Distortion Matrix:---\n {self.dist}")
        print(f"---Rotation Vectors:---\n {self.rvecs}")
        print(f"---Translation Vectors:---\n {self.tvecs}")

    # TODO Fix Settings Saving and Loading
    # Gespeicherte Einstellungen laden // Load saved settings
    def load_settings(self, data_folder, filename="camera_settings.json"):
        with open(data_folder / filename) as json_file:
            data = json.load(json_file)
            self.f = data["f"]
            self.cameraMatrix = np.asarray(data["cameraMatrix"])
            self.newCameramatrix = np.asarray(data["newCameramatrix"])
            self.dist = np.array(data["dist"])
            print(data)
        print("---Kamera Parameter geladen---")

    # Kamera Parameter speichern // Save camera parameters
    def save_settings(self, data_folder, filename="camera_settings.json"):
        settings = {
            "f": self.f,
            "cameraMatrix": self.cameraMatrix.tolist(),
            "newCameramatrix": self.newCameramatrix.tolist(),
            "dist": self.dist.tolist(),
        }

        json_obj = json.dumps(settings, indent=4)

        with open(data_folder / filename, "w") as outfile:
            outfile.write(json_obj)

        print("---Kamera Parameter gespeichert---")

    def calibrate_charuco(self, folder, charboard, dictionary, calibfilesdtype="jpg"):
        arucodict = cv.aruco.getPredefinedDictionary(dictionary)
        params = cv.aruco.DetectorParameters()
        charparams = cv.aruco.CharucoParameters()
        refinedparams = cv.aruco.RefineParameters()


        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

        ardetector = cv.aruco.ArucoDetector(arucodict, params)
        chardetector = cv.aruco.CharucoDetector(charboard, charparams, params)
        objPoints = []
        imgPoints = []
        all_charuco_corners = []
        all_charuco_ids = []
        # Bilder für Kalibrierung // Calibration images
        size=()
        images = list(folder.glob(f"*.{calibfilesdtype}"))
        for image in images:
            img = cv.imread(image)
            size = img.shape[:2]
            image_copy = img.copy()
            marker_corners, marker_ids, rejected_corners = ardetector.detectMarkers(img)
            if len(marker_ids) > 0:
                cv.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)

                marker_corners, marker_ids, _, _ = ardetector.refineDetectedMarkers(img, charboard, marker_corners, marker_ids, rejected_corners)
                charuco_corners, charuco_ids, marker_corners, marker_ids = chardetector.detectBoard(img, markerCorners=marker_corners, markerIds=marker_ids)
                imgpts, objpts = charboard.matchImagePoints(charuco_corners, charuco_ids)
                if len(charuco_ids)>0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    objPoints.append(objpts)
                    imgPoints.append(imgpts)
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, size, None, None)

        if retval:
            self.cameraMatrix = camera_matrix
            self.dist = dist_coeffs
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.f = self.cameraMatrix[0, 0]
            self.imgPoints = imgPoints
            self.objPoints = objPoints



