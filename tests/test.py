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
DEBUG = False
#PathHandler
handler = PathHandler.PathHandler()
#testimage
testimage = "image_15.jpg"
#ImageProcessor
processor = ImageProcessor.ImageProcessor(debug=DEBUG)
#ArucoDict

#Color HSV Ranges


newercolors = [
    (np.array([50.0, int(255*0.3), int(255*0.25)]), np.array([66.0, int(255*0.7), int(255*0.75)]), "green"),
    #(np.array([100.0, int(255*0.04), int(255*0.19)]), np.array([115.0, int(255*0.2), int(255*0.39)]), "black"),
    (np.array([0.0, int(255*0.25), int(255*0.3)]), np.array([180.0, int(255*0.75), int(255*0.79)]), "red"),
    (np.array([83.0, int(255*0.1), int(255*0.30)]), np.array([99.0, int(255*0.4), int(255*0.79)]), "blue"),
    (np.array([31.0, int(255*0.25), int(255*0.3)]), np.array([39.0, int(255*0.65), int(255*0.85)]), "yellow"),
    (np.array([0.0, int(255*0.15), int(255*0.4)]), np.array([180.0, int(255*0.4), int(255*0.8)]), "pink")
]

def pi_stereo():
    pi1 = Camera.Camera(debug=DEBUG)
    pi2 = Camera.Camera(debug=DEBUG)

    handler.set_calibration_images_folder("example_data/new_calibration")
    handler.set_image_folder("example_data/new_images")

    board = processor.create_charuco((9, 6), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    board2 = processor.create_charuco((13, 7), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    handler.set_calibration_images_folder("example_data/new_calibration_charuco")
    pi1.calibrate_charuco(handler.CALIB_FOLDER / "calibration_left", board, board2, cv.aruco.DICT_5X5_100)
    pi2.calibrate_charuco(handler.CALIB_FOLDER / "calibration_right", board, board2, cv.aruco.DICT_5X5_100)
    #pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    #pi2.calibrate(handler.CALIB_FOLDER / "calibration_right", RESOLUTION, (13, 9))
    #pi1.save_settings(SETTINGS_FOLDER / "pi1")
    #pi2.save_settings(SETTINGS_FOLDER / "pi2")
    #pi1.load_settings(SETTINGS_FOLDER / "pi1")
    #pi2.load_settings(SETTINGS_FOLDER / "pi2")
    picam = StereoCamera.StereoCamera(pi1, pi2, 0.05, 60.0)
    img_left = cv.imread(handler.IMAGE_FOLDER /"new_images_left" / testimage)
    img_right = cv.imread(handler.IMAGE_FOLDER / "new_images_right" / testimage)
    #img_left = processor.rotate(processor.rotate(img_left))
    #img_right = processor.rotate(processor.rotate(img_right))


    picam.stereo_calibration_rectification(img_left.shape[:-1][::-1])
    picam.save_map(handler.SETTINGS_FOLDER / "stereo")
    #picam.read_map(handler.SETTINGS_FOLDER / "stereo")
    stereoproc = StereoProcessor(debug=DEBUG)
    # Image undistortion Rectification
    undistorted_left, undistorted_right = stereoproc.undistort_rectify(img_left, img_right, picam)

    undistorted_left_c = cv.resize(undistorted_left.copy(), (1800, 1000))
    undistorted_right_c = cv.resize(undistorted_right.copy(), (1800, 1000))
    processor.showimg(undistorted_left_c, "left")
    processor.showimg(undistorted_right_c, "right")
    #undistorted_left, undistorted_right = undistored1, undistored2


    masks = stereoproc.combine_masks(processor.createmasks(undistorted_left, newercolors), processor.createmasks(undistorted_right, newercolors))
    framer, framel, pr, pl = stereoproc.cons_stereo(undistorted_left, undistorted_right, masks)

    markercorners, markerids = processor.detect_aruco(framel, cv.aruco.DICT_6X6_50)

    if len(pr) == len(pl):
        for i in range(len(pr)):


            dist = stereoproc.find_depth(pr[i][0], pl[i][0], undistorted_left, undistorted_right, picam)
            pos = stereoproc.calcdist(markercorners[0], markercorners[1], pl[i][0]. pl[i][1])
            print(pos)


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