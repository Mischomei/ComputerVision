import threading

from src.PiCamera.Picamera import Picamera

cam1 = Picamera(1)
cam2 = Picamera(2)


stream1 = threading.Thread(cam1.preview())
stream2 = threading.Thread(cam2.preview())
capture1 = threading.Thread()
stream1.start()
stream2.start()

threading.Timer(60.0, stream1.join)
threading.Timer(60.0, stream2.join)

exit()