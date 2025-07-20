import os
from pathlib import Path

import numpy as np
import cv2 as cv

from src.ImageProcessing import ImageProcessor
from src.Camera import StereoCamera, Camera
from src.ImageProcessing.StereoProcessor import StereoProcessor
from src.extras import PathHandler

#TODO Dynamic Image Pipeline and Path declarations
#Paths

container_points = np.array([[0, 0, 0], [0.074, 0, 0], [0.074, 0, 0.045], [0.074, 0.037, 0.045], [0, 0.037, 0.045], [0, 0.037, 0]], dtype=np.float64)
container_points2 = np.array([[0, 0, 0], [0, 0.037, 0], [0, 0.037, 0.045], [0.074, 0.037, 0.045], [0.074, 0, 0.045], [0.074, 0, 0]], dtype=np.float64)
#Resolution
RESOLUTION = (1920, 1080)
#Debug
DEBUG = True
#PathHandler
handler = PathHandler.PathHandler()
#Reference image for histogram matching
refimage = "image_16.jpg"
#testimage
testimage = "image_15.jpg"
#ImageProcessor
processor = ImageProcessor.ImageProcessor(debug=DEBUG)
#ArucoDict

#Color HSV Ranges




newercolors = [
    (np.array([50.0, int(255*0.25), int(255*0.2)]), np.array([74.0, int(255*0.9), int(255*0.8)]), "green"),
    #(np.array([100.0, int(255*0.04), int(255*0.19)]), np.array([115.0, int(255*0.2), int(255*0.39)]), "black"),
    (np.array([[160.0, int(255*0.35), int(255*0.5)],[0.0, int(255*0.25), int(255*0.3)]]), np.array([[180.0, int(255*0.9), int(255*0.9)], [3.5, int(255*0.9), int(255*0.9)]]), "red"),
    (np.array([85.5, int(255*0.25), int(255*0.3)]), np.array([97.0, int(255*0.55), int(255*0.8)]), "blue"),
    (np.array([32.0, int(255*0.3), int(255*0.35)]), np.array([39.5, int(255*0.7), int(255*0.85)]), "yellow"),
    (np.array([[174.0, int(255*0.1), int(255*0.35)], [0.0, int(255*0.1), int(255*0.35)]]), np.array([[180.0, int(255*0.45), int(255*0.85)], [2.0, int(255*0.45), int(255*0.85)]]), "pink")
]
newercolors = [
    (np.array([52.0, 80, 0]), np.array([86.0, 255, 255]), "green"),
    #(np.array([100.0, int(255*0.04), int(255*0.19)]), np.array([115.0, int(255*0.2), int(255*0.39)]), "black"),
    (np.array([[167.0, 100, 0],[0.0, 20, 0]]), np.array([[180.0, 255, 255], [7.0, 255, 255]]), "red"),
    (np.array([86.0, 60, 0]), np.array([115.0, 255, 255]), "blue"),
    (np.array([28.0, 55, 0]), np.array([78.0, 160, 255]), "yellow"),
    (np.array([170.0, 0, 0]), np.array([180.0, 255, 255]), "pink")
]




def pi_stereo():
    pi1 = Camera.Camera(debug=DEBUG)
    pi2 = Camera.Camera(debug=DEBUG)

    handler.set_calibration_images_folder("example_data/new_calibration")
    handler.set_image_folder("example_data/new_images")

    #Calibration

    #board = processor.create_charuco((9, 6), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    #board2 = processor.create_charuco((13, 7), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    #handler.set_calibration_images_folder("example_data/new_calibration_charuco")
    #pi1.calibrate_charuco(handler.CALIB_FOLDER / "calibration_left", board, board2, cv.aruco.DICT_5X5_100)
    #pi2.calibrate_charuco(handler.CALIB_FOLDER / "calibration_right", board, board2, cv.aruco.DICT_5X5_100)
    #pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    #pi2.calibrate(handler.CALIB_FOLDER / "calibration_right", RESOLUTION, (13, 9))
    #pi1.save_settings(handler.SETTINGS_FOLDER / "pi1")
    #pi2.save_settings(handler.SETTINGS_FOLDER / "pi2")


    pi1.load_settings(handler.SETTINGS_FOLDER / "pi1")
    pi2.load_settings(handler.SETTINGS_FOLDER / "pi2")
    picam = StereoCamera.StereoCamera(pi1, pi2, 0.05, 60.0)

    reference_left = cv.imread(handler.IMAGE_FOLDER / "new_images_left" / refimage)
    reference_right = cv.imread(handler.IMAGE_FOLDER / "new_images_right" / refimage)
    #reference_left = processor.crop(reference_left, 63, 2390, 892, 892+2946)
    #reference_right = processor.crop(reference_right, 63, 2390, 892, 892 + 2946)



    img_left = cv.imread(handler.IMAGE_FOLDER /"new_images_left" / testimage)
    img_right = cv.imread(handler.IMAGE_FOLDER / "new_images_right" / testimage)
    #img_left = processor.rotate(processor.rotate(img_left))
    #img_right = processor.rotate(processor.rotate(img_right))


    #Histogram matching


    #picam.stereo_calibration_rectification(img_left.shape[:-1][::-1])
    #picam.save_map(handler.SETTINGS_FOLDER / "stereo")
    picam.read_map(handler.SETTINGS_FOLDER / "stereo")
    stereoproc = StereoProcessor(debug=DEBUG)
    # Image undistortion Rectification
    histmatched_left = processor.histogram_matching(reference_left, img_left)
    histmatched_right = processor.histogram_matching(reference_right, img_right)
    undistorted_left, undistorted_right = stereoproc.undistort_rectify(img_left, img_right, picam)
    histmatched_left_undistorted, histmatched_right_undistorted = stereoproc.undistort_rectify(histmatched_left, histmatched_right, picam)


    undistorted_left_c = cv.resize(histmatched_left_undistorted.copy(), (1800, 1000))
    undistorted_right_c = cv.resize(histmatched_right_undistorted.copy(), (1800, 1000))
    processor.showimg(undistorted_left_c, "left")
    processor.showimg(undistorted_right_c, "right")
    #undistorted_left, undistorted_right = undistored1, undistored2

    #Histogram matching

    cv.imwrite(handler.SAVE_FOLDER/"histmatched_left.jpg", histmatched_left_undistorted)
    cv.imwrite(handler.SAVE_FOLDER/"histmatched_right.jpg", histmatched_right_undistorted)

    masks = stereoproc.combine_masks(processor.createmasks(histmatched_left_undistorted, newercolors), processor.createmasks(histmatched_right_undistorted, newercolors))
    framer, framel, pr, pl = stereoproc.cons_stereo(undistorted_left, undistorted_right, masks)

    markercorners, markerids = processor.detect_aruco(framel, cv.aruco.DICT_6X6_50)




    framel = cv.resize(framel, (1800, 1000))
    processor.showimg(framel, "framel")
    framer = cv.resize(framer, (1800, 1000))
    processor.showimg(framer, "framer")
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_left.jpg"), framel)
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_right.jpg"), framer)
    #stereoproc.find_depth(p2, p1, undistorted_left, undistorted_right, picam)

def tryingPnP():
    #pi1 = Camera.Camera(debug=DEBUG)

    #handler.set_calibration_images_folder("example_data/new_calibration")
    handler.set_image_folder("example_data/new_images")

    #pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    board = processor.create_charuco((9, 6), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    pi1 = Camera.Camera()
    handler.set_calibration_images_folder("example_data/new_calibration_charuco")
    pi1.calibrate_charuco(handler.CALIB_FOLDER / "calibration_left", board, cv.aruco.DICT_5X5_100)

    img_left = cv.imread(handler.IMAGE_FOLDER /"new_images_left" / testimage)
    undistored1 = processor.undistort(handler.IMAGE_FOLDER / "new_images_left", testimage, pi1, handler.SAVE_FOLDER)
    undistored1 = processor.rotate(processor.rotate(undistored1))

    masks = processor.createmasks(undistored1, newercolors)
    masks = processor.maskstolist(masks)
    outimg = undistored1.copy()

    usedpolys = []

    for mask in masks:


        edges = processor.get_color_channels(undistored1, mask[0])
        for poly in usedpolys:
            edges = cv.fillPoly(edges, [poly], (0, 0, 0))
        polyn, points = processor.lines(edges, 0.05)

        outimg = processor.drawcontour(outimg, points, polyn, mask[1])

        ret, aruco_rvec, aruco_tvec = processor.aruco_pose_estimation(undistored1, cv.aruco.DICT_6X6_50, 0.051, pi1.cameraMatrix, pi1.dist-pi1.dist)

        if ret:


            outimg =processor.pose_drawing(outimg, aruco_rvec, aruco_tvec,  0.051, pi1.cameraMatrix, pi1.dist - pi1.dist)
        ret = False
        if len(points) == 6:
            ret, rvec, tvec = processor.trypnp(undistored1, container_points, points.astype(np.float64), pi1.cameraMatrix, pi1.dist-pi1.dist, 0.1)


        if ret:
            outimg = processor.pose_drawing(outimg, [rvec], [tvec], 0.04, pi1.cameraMatrix, pi1.dist - pi1.dist)

            #d = (dist / normdist) * 0.385
            #print(d)
        usedpolys.append(points.astype(np.int32))

    outimg = cv.resize(outimg, (1800, 1000))
    processor.showimg(outimg, "tryingPnP")
    cv.imwrite(handler.SAVE_FOLDER/"testoutimg.jpg", outimg)

def test_markers():
    processor.generate_aruco(handler.SAVE_FOLDER, cv.aruco.DICT_6X6_50, 4, 600, 16)

if __name__ == "__main__":
    #board = processor.create_charuco((9, 6), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    #board2 = processor.create_charuco((13, 7), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)

    #pi1 = Camera.Camera()
    #handler.set_calibration_images_folder("example_data/new_calibration_charuco")
    #pi1.calibrate_charuco(handler.CALIB_FOLDER / "calibration_left", board, board2, cv.aruco.DICT_5X5_100)

    pi_stereo()
    #tryingPnP()