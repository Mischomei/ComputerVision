import glob
import cv2 as cv
import numpy as np
import os
import json
import time

# Kamera Klasse
class Camera:
    #Kameraparameter // Camera parameters
    f = None # Focal length
    resolution = (0, 0) # Auflösung // Resolution
    cameraMatrix, dist, rvecs, tvecs = None, None, None ,None # Kameramatrix, Verzerrungsmatrix, Rotationsvektoren, Verschiebungsvektoren // Camera Matrix, Distortion Matrix, Rotation Vectors, Translation Vectors

    def __init__(self, resolution = (0,0), data=None):
        self.resolution = resolution
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
        images = glob.glob(folder + f'/*.{calibfilesdtype}')
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
                cv.imshow("Img", img)
                cv.waitKey(1000)

        cv.destroyAllWindows()

        # Kalibrierung // Calibration

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, res, None, None)

        if ret:
            self.cameraMatrix = cameraMatrix
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.f = cameraMatrix[0, 0]
            print(f"---Kamera kalibriert: {ret}---")
        else:
            print(f"---Kalibrierung fehlgeschlagen---")

        return ret


    def show_parameters(self):
        print(f"---Focal Length: {self.f}---")
        print(f"---Resolution: {self.resolution}---")
        print(f"---Camera Matrix:---\n {self.cameraMatrix}")
        print(f"---Distortion Matrix:---\n {self.dist}")
        print(f"---Rotation Vectors:---\n {self.rvecs}")
        print(f"---Translation Vectors:---\n {self.tvecs}")

    # Gespeicherte Einstellungen laden // Load saved settings
    def load_settings(self, data_folder, filename="camera_settings.json"):
        with open(os.path.join(data_folder, filename)) as json_file:
            data = json.load(json_file)
            self.f = data["f"]
            self.resolution = data["resolution"]
            self.cameraMatrix = np.array(data["cameraMatrix"])
            self.dist = np.array(data["dist"])
            self.rvecs = np.array(data["rvecs"])
            self.tvecs = np.array(data["tvecs"])
            print(data)
        print("---Kamera Parameter geladen---")

    # Kamera Parameter speichern // Save camera parameters
    def save_settings(self, data_folder, filename="camera_settings.json"):
        settings = {
            "f": self.f,
            "resolution": self.resolution,
            "cameraMatrix": self.cameraMatrix.tolist(),
            "dist": self.dist.tolist(),
            "rvecs": np.array(self.rvecs).tolist(),
            "tvecs": np.array(self.tvecs).tolist()
        }

        json_obj = json.dumps(settings, indent=4)

        with open(os.path.join(data_folder, filename), "w") as outfile:
            outfile.write(json_obj)

        print("---Kamera Parameter gespeichert---")

    #Livevideo der Kamera mit möglichkeit Photos zu machen // Livevideo of camera with the possibility to take photos
    def start_stream(self, camnum = 0, save_folder = "images/stream"):
        cap = cv.VideoCapture(camnum)
        while True:
            ret, frame = cap.read()
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("---Stream beendet---")
                break
            if cv.waitKey(1) & 0xFF == ord('s'):
                cv.imwrite(os.path.join(save_folder, f"stream_{int(time.time())}.jpg"), frame)
                print("---Frame gespeichert---")
        cap.release()
        cv.destroyAllWindows()
