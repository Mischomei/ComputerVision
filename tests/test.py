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
#testimage
testimage = "image_4.jpg"
#ImageProcessor
processor = ImageProcessor.ImageProcessor(debug=DEBUG)
#ArucoDict

#Color HSV Ranges
colors = [
        (np.array([80.0, 20, 35]), np.array([100.0, 255, 255]), "green"),
        (np.array([125.0, 110, 50]), np.array([160.0, 170, 230]), "red")
              ]

newcolors = [
    (np.array([87.0, int(255*0.7), int(255*0.4)]), np.array([100.0, 255, int(255*0.78)]), "green"),
    (np.array([110.0, int(255*0.45), int(255*0.3)]), np.array([113.5, int(255*0.82), int(255*0.62)]), "black"),
    (np.array([130.0, int(255*0.5), int(255*0.45)]), np.array([170.0, int(255*0.82), int(255*0.85)]), "red"),
    (np.array([88.5, int(255*0.5), int(255*0.35)]), np.array([97.0, int(255), int(255)]), "cyan")
]

def pi_stereo():
    pi1 = Camera.Camera(debug=DEBUG)
    pi2 = Camera.Camera(debug=DEBUG)

    handler.set_calibration_images_folder("example_data/new_calibration")
    handler.set_image_folder("example_data/new_images")

    pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    pi2.calibrate(handler.CALIB_FOLDER / "calibration_right", RESOLUTION, (13, 9))
    #pi1.save_settings(SETTINGS_FOLDER / "pi1")
    #pi2.save_settings(SETTINGS_FOLDER / "pi2")
    #pi1.load_settings(SETTINGS_FOLDER / "pi1")
    #pi2.load_settings(SETTINGS_FOLDER / "pi2")
    picam = StereoCamera.StereoCamera(pi1, pi2, 5, 60.0)
    img_left = cv.imread(handler.IMAGE_FOLDER /"new_images_left" / testimage)
    img_right = cv.imread(handler.IMAGE_FOLDER / "new_images_right" / testimage)
    undistored1 = processor.undistort(handler.IMAGE_FOLDER / "new_images_left", testimage, pi1, handler.SAVE_FOLDER)
    undistored2 = processor.undistort(handler.IMAGE_FOLDER / "new_images_right", testimage, pi2, handler.SAVE_FOLDER)
    processor.showimg(undistored1, "u1")
    processor.showimg(undistored2, "u2")
    img_left = processor.rotate(processor.rotate(img_left))
    img_right = processor.rotate(processor.rotate(img_right))


    picam.stereo_calibration_rectification(img_left.shape[:-1][::-1])
    picam.save_map(handler.SETTINGS_FOLDER / "stereo")
    #picam.read_map(handler.SETTINGS_FOLDER / "stereo")
    stereoproc = StereoProcessor(debug=DEBUG)
    # Image undistortion Rectification
    #img_left = cv.imread(IMAGE_FOLDER / "container_left" / testimage)
    undistorted_left, undistorted_right = stereoproc.undistort_rectify(img_left, img_right, picam)
    #undistorted_left = processor.crop(undistorted_left, 300, 900, 400, 1400)
    #undistorted_right = processor.crop(undistorted_right, 350, 1050, 300, 1300)

    processor.showimg(undistorted_left, "left")
    processor.showimg(undistorted_right, "right")
    #undistorted_left, undistorted_right = undistored1, undistored2


    masks = stereoproc.combine_masks(processor.createmasks(undistorted_left, newcolors), processor.createmasks(undistorted_right, newcolors))
    framer, framel = stereoproc.cons_stereo(undistorted_left, undistorted_right, masks)
    framel = cv.resize(framel, (1000, 1000))
    processor.showimg(framel, "framel")
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

    handler.set_calibration_images_folder("example_data/new_calibration")
    handler.set_image_folder("example_data/new_images")

    pi1.calibrate(handler.CALIB_FOLDER / "calibration_left", RESOLUTION, (13, 9))
    img_left = cv.imread(handler.IMAGE_FOLDER /"new_images_left" / testimage)
    img_left = processor.crop(img_left, 100, 2300, 0, 2700)
    img_left = processor.rotate(processor.rotate(img_left))
    undistored1 = processor.undistort(handler.IMAGE_FOLDER / "new_images_left", testimage, pi1, handler.SAVE_FOLDER)
    undistored1 = processor.crop(undistored1, 100, 2300, 0, 2700)
    undistored1 = processor.rotate(processor.rotate(undistored1))
    undistored1 = cv.resize(undistored1, (1200, 978))
    contouredimg = undistored1.copy()
    masks = processor.createmasks(undistored1, newcolors)
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

        ret, rvec, tvec = processor.trypnp(undistored1, container_points, points.astype(np.float64), pi1.cameraMatrix, pi1.dist-pi1.dist, 0.1)


        if ret:
            outimg = processor.pose_drawing(outimg, [rvec], [tvec], 0.04, pi1.cameraMatrix, pi1.dist - pi1.dist)

            #d = (dist / normdist) * 0.385
            #print(d)
        usedpolys.append(points.astype(np.int32))

    processor.showimg(outimg, "tryingPnP")

def test_markers():
    processor.generate_aruco(handler.SAVE_FOLDER, cv.aruco.DICT_6X6_50, 4, 600, 16)

if __name__ == "__main__":
    processor.create_charuco((13, 7), 0.05, 0.04, cv.aruco.DICT_5X5_100, handler.SAVE_FOLDER)
    #pi_stereo()
    #tryingPnP()