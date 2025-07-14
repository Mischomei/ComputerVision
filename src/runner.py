from pathlib import Path
import sys


import numpy as np
import cv2 as cv
sys.path.insert(1, (Path.cwd().parent).as_posix())

from ImageProcessing import ImageProcessor
from Camera import StereoCamera, Camera
from PiCamera.Picamera import Picamera

from ImageProcessing.StereoProcessor import StereoProcessor
from extras import PathHandler

import os

container_points = np.array([[0, 0, 0], [0.074, 0, 0], [0.074, 0, 0.045], [0.074, 0.037, 0.045], [0, 0.037, 0.045], [0, 0.037, 0]], dtype=np.float64)
container_points2 = np.array([[0, 0, 0], [0, 0.037, 0], [0, 0.037, 0.045], [0.074, 0.037, 0.045], [0.074, 0, 0.045], [0.074, 0, 0]], dtype=np.float64)
#Resolution
RESOLUTION = (1920, 1080)
#Debug
DEBUG = True
#PathHandler
handler = PathHandler.PathHandler()
#testimage
testimage = "image_15.jpg"
#ImageProcessor
processor = ImageProcessor.ImageProcessor(debug=DEBUG)
#ArucoDict
stereoproc = StereoProcessor(debug=DEBUG)

handler.set_calibration_images_folder("example_data/new_calibration_charuco")
#handler.set_calibration_images_folder("example_data/new_calibration_aruco")
handler.set_image_folder("example_data/new_images")
cam_right = Picamera(0)
cam_left = Picamera(1)

newercolors = [
    (np.array([50.0, int(255*0.3), int(255*0.25)]), np.array([66.0, int(255*0.7), int(255*0.75)]), "green"),
    #(np.array([100.0, int(255*0.04), int(255*0.19)]), np.array([115.0, int(255*0.2), int(255*0.39)]), "black"),
    (np.array([0.0, int(255*0.25), int(255*0.3)]), np.array([180.0, int(255*0.75), int(255*0.79)]), "red"),
    (np.array([83.0, int(255*0.1), int(255*0.30)]), np.array([99.0, int(255*0.4), int(255*0.79)]), "blue"),
    (np.array([31.0, int(255*0.25), int(255*0.3)]), np.array([39.0, int(255*0.65), int(255*0.85)]), "yellow"),
    (np.array([0.0, int(255*0.15), int(255*0.4)]), np.array([180.0, int(255*0.4), int(255*0.8)]), "pink")
]

pi1 = Camera.Camera(debug=DEBUG)
pi2 = Camera.Camera(debug=DEBUG)
picam = StereoCamera.StereoCamera(pi1, pi2, 0.05, 60.0)

handler.set_calibration_images_folder("example_data/new_calibration")
handler.set_image_folder("example_data/new_images")


def calibrate():
    board = processor.create_charuco((9, 6), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    board2 = processor.create_charuco((13, 7), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    handler.set_calibration_images_folder("example_data/new_calibration_charuco")
    pi1.calibrate_charuco(handler.CALIB_FOLDER / "calibration_left", board, board2, cv.aruco.DICT_5X5_100)
    pi2.calibrate_charuco(handler.CALIB_FOLDER / "calibration_right", board, board2, cv.aruco.DICT_5X5_100)
    picam.stereo_calibration_rectification((4500, 2500))
    picam.save_map(handler.SETTINGS_FOLDER / "stereo")


def initialize():
    cam_right.preview()
    cam_right.start()
    cam_left.preview()
    cam_left.start()

def img_recognition():
    name = "image_final.jpg"
    pic1 = cam_right.capture_normal(0, handler.IMAGE_FOLDER / "new_images_right" / name)
    pic2 = cam_left.capture_normal(0, handler.IMAGE_FOLDER / "new_images_left" / name)

    img_left = cv.imread(handler.IMAGE_FOLDER / "new_images_left" / name)
    img_right = cv.imread(handler.IMAGE_FOLDER / "new_images_right" / name)

    undistorted_left, undistorted_right = stereoproc.undistort_rectify(img_left, img_right, picam)

    masks = stereoproc.combine_masks(processor.createmasks(undistorted_left, newercolors),
                                     processor.createmasks(undistorted_right, newercolors))
    framer, framel, pr, pl = stereoproc.cons_stereo(undistorted_left, undistorted_right, masks)

    markercorners, markerids = processor.detect_aruco(framel, cv.aruco.DICT_6X6_50)
    dists = []
    if len(pr) == len(pl):

        for i in range(len(pr)):
            dist = stereoproc.find_depth(pr[i][0], pl[i][0], undistorted_left, undistorted_right, picam)
            pos = stereoproc.calcdist(markercorners[0], markercorners[1], pl[i][0].pl[i][1])
            print(pos)
            dists.append(pos)
    framel = cv.resize(framel, (1800, 1000))
    processor.showimg(framel, "framel")
    framer = cv.resize(framer, (1800, 1000))
    processor.showimg(framer, "framer")
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_left.jpg"), framel)
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_right.jpg"), framer)
    return dists