import threading
import sys
import time
from pathlib import Path
import curses
sys.path.insert(1, (Path.cwd().parent).as_posix())
from src.PiCamera.Picamera import Picamera
from src.ImageProcessing.ImageProcessor import ImageProcessor
import cv2 as cv
from src.extras.CursesHandler import CursesHandler
from src.extras.PathHandler import PathHandler


handler = PathHandler()
handler.set_calibration_images_folder("example_data/new_calibration")
proc = ImageProcessor()
cam_right = Picamera(0)

cam_left = Picamera(1)

cam_right.preview()
cam_right.start()
cam_left.preview()
cam_left.start()

def calibration():
    num = 0
    with CursesHandler() as stdsrc:
        while True:
            press = stdsrc.getkey()
            if press == "s":
                pic1 = cam_right.capture_normal(0, handler.CALIB_FOLDER / "calibration_right" / f"calibration_{num}.jpg")
                pic2 = cam_left.capture_normal(0, handler.CALIB_FOLDER / "calibration_left" / f"calibration_{num}.jpg")
                proc.showimg(pic1, f"calibration_left_{num}")
                proc.showimg(pic2, f"calibration_right_{num}")
                num +=1
            if press == "q":
                break

def take_pictures(num = 0):
    with CursesHandler() as stdsrc:
        while True:
            press = stdsrc.getkey()
            if press == "s":
                pic1 = cam_right.capture_normal(0, handler.IMAGE_FOLDER / "mew_images_right" / f"image_{num}.jpg")
                pic2 = cam_left.capture_normal(0, handler.IMAGE_FOLDER / "new_images_left" / f"image_{num}.jpg")
                proc.showimg(pic1, f"new_picture_{num}")
                proc.showimg(pic2, f"new_picture{num}")
                num +=1
            if press == "q":
                break

def test():
    with CursesHandler() as stdsrc:
        while True:
            press = stdsrc.getkey()
            if press == "s":
                pic = cam_right.capture_normal(1)
                proc.showimg(pic, "picutre")
            if press == "q":
                break

take_pictures(0)

#stream1 = threading.Thread(cam1.preview())
#stream2 = threading.Thread(cam2.preview())
#capture1 = threading.Thread()
#stream1.start()
#stream2.start()

#threading.Timer(60.0, stream1.join)

#threading.Timer(60.0, stream2.join)
