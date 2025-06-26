import threading
import sys
import time
from pathlib import Path
import curses
sys.path.insert(1, (Path.cwd().parent).as_posix())
from src.PiCamera.Picamera import Picamera
from src.ImageProcessing.ImageProcessor import ImageProcessor
import cv2 as cv

stdsrc = curses.initscr()

proc = ImageProcessor()
cam1 = Picamera(0)

cam2 = Picamera(1)

cam1.preview()
cam1.start()
cam2.preview()
cam2.start()

while True:
    press = stdsrc.getkey()
    if press == "s":
        pic = cam1.capture_normal(0)
        proc.showimg(pic, "picutre")
    if press == "q":
        break



#stream1 = threading.Thread(cam1.preview())
#stream2 = threading.Thread(cam2.preview())
#capture1 = threading.Thread()
#stream1.start()
#stream2.start()

#threading.Timer(60.0, stream1.join)

#threading.Timer(60.0, stream2.join)
