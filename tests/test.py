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
container_points2 = np.array([[0, 0, 0], [0.074, 0, 0], [0.074, 0, 0.045], [0.074, 0.037, 0.045], [0, 0.037, 0.045], [0, 0.037, 0]], dtype=np.float64)
#Resolution
RESOLUTION = (1920, 1080)
#Debug
DEBUG = True
#PathHandler
handler = PathHandler.PathHandler()
#testimage
testimage = "testcontainer2.jpg"
#ImageProcessor
processor = ImageProcessor.ImageProcessor(debug=DEBUG)
#ArucoDict

#Color HSV Ranges
colors = [
        (np.array([80.0, 20, 35]), np.array([100.0, 255, 255]), "green"),
        (np.array([125.0, 110, 50]), np.array([160.0, 170, 230]), "red")
              ]

def pi_stereo():
    pi1 = Camera.Camera(debug=DEBUG)
    pi2 = Camera.Camera(debug=DEBUG)
    pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    pi2.calibrate(handler.CALIB_FOLDER / "calibration_right", RESOLUTION, (13, 9))
    #pi1.save_settings(SETTINGS_FOLDER / "pi1")
    #pi2.save_settings(SETTINGS_FOLDER / "pi2")
    #pi1.load_settings(SETTINGS_FOLDER / "pi1")
    #pi2.load_settings(SETTINGS_FOLDER / "pi2")
    picam = StereoCamera.StereoCamera(pi1, pi2, 5, 60.0)
    img_left = cv.imread(handler.IMAGE_FOLDER /"container_left" / testimage)
    img_right = cv.imread(handler.IMAGE_FOLDER / "container_right" / testimage)
    undistored1 = processor.undistort(handler.IMAGE_FOLDER / "container_left", testimage, pi1, handler.SAVE_FOLDER)
    undistored2 = processor.undistort(handler.IMAGE_FOLDER / "container_right", testimage, pi2, handler.SAVE_FOLDER)
    processor.showimg(undistored1, "u1")
    processor.showimg(undistored2, "u2")
    undistored1 = processor.rotate(processor.rotate(img_left))
    undistored2 = processor.rotate(processor.rotate(img_right))


    #picam.stereo_calibration_rectification(undistored1.shape[:-1][::-1])
    #picam.save_map(SETTINGS_FOLDER / "stereo")
    picam.read_map(handler.SETTINGS_FOLDER / "stereo")
    stereoproc = StereoProcessor(debug=DEBUG)
    # Image undistortion Rectification
    #img_left = cv.imread(IMAGE_FOLDER / "container_left" / testimage)
    undistorted_left, undistorted_right = stereoproc.undistort_rectify(undistored1, undistored2, picam)
    undistorted_left = processor.crop(undistorted_left, 300, 900, 400, 1400)
    undistorted_right = processor.crop(undistorted_right, 350, 1050, 300, 1300)

    processor.showimg(undistorted_left, "left")
    processor.showimg(undistorted_right, "right")
    #undistorted_left, undistorted_right = undistored1, undistored2


    masks = stereoproc.combine_masks(processor.createmasks(undistorted_left, colors), processor.createmasks(undistorted_right, colors))
    framer, framel = stereoproc.cons_stereo(undistorted_left, undistorted_right, masks)
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_left.jpg"), framel)
    cv.imwrite(os.path.join(handler.SAVE_FOLDER.as_posix(), "frame_right.jpg"), framer)
    #stereoproc.find_depth(p2, p1, undistorted_left, undistorted_right, picam)


def webcam():
    cam = Camera.Camera(debug=DEBUG)
    cam.calibrate(handler.DATA_PATH / "images/calibration_laptopweb1", (1280, 720), (13, 9))
    cam.save_settings(handler.SETTINGS_FOLDER / "webcam1")
    capt = cv.VideoCapture(0)
    while True:
        ret, frame = capt.read()
        if ret:
            ret2, r_vecs, t_vecs = processor.aruco_pose_estimation(frame, cv.aruco.DICT_6X6_50, 0.2, cam.cameraMatrix, cam.dist)
            if ret2:
                frame = processor.pose_drawing(frame, r_vecs, t_vecs, 0.2, cam.cameraMatrix, cam.dist)
            cv.imshow("Aruco", frame)

        if cv.waitKey(1) == ord('q'):
            break
    capt.release()
    cv.destroyAllWindows()


def tryingPnP():
    pi1 = Camera.Camera(debug=DEBUG)
    pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    img_left = cv.imread(handler.IMAGE_FOLDER /"container_left" / testimage)
    img_left = processor.rotate(processor.rotate(img_left))
    undistored1 = processor.undistort(handler.IMAGE_FOLDER / "container_left", testimage, pi1, handler.SAVE_FOLDER)
    undistored1 = processor.rotate(processor.rotate(undistored1))
    contouredimg = undistored1.copy()
    masks = processor.createmasks(undistored1, colors)
    masks = processor.maskstolist(masks)
    for mask in masks:
        edges = processor.get_color_channels(undistored1, mask[0])
        polyn, points = processor.lines(edges, 0.05)
        contouredimg = processor.drawcontour(contouredimg, points, polyn, mask[1])
        ret = False
        ret, rvec, tvec = processor.trypnp(undistored1, container_points, points.astype(np.float64), pi1.cameraMatrix, pi1.dist, 0.1)
        if ret:
            newcontouredimg = processor.pose_drawing(undistored1, [rvec], [tvec], 0.04, pi1.cameraMatrix, pi1.dist - pi1.dist)
            processor.showimg(newcontouredimg, "tryingPnP_contoured")
    processor.showimg(contouredimg, "tryingPnP")

def test_markers():
    processor.generate_aruco(handler.SAVE_FOLDER, cv.aruco.DICT_6X6_50, 4, 600, 16)

if __name__ == "__main__":
    tryingPnP()