from picamera2 import Picamera2 , Preview

CAPTURE_JPG = 0
CAPTURE_ARRAY = 1

class Picamera:
    camnum: int
    picam: Picamera2
    preview_config = 0
    video_config = 0
    def __init__(self, cam_num):
        self.camnum = cam_num
        self.picam = Picamera2()
        self.preview_config = self.picam.create_preview_configuration(main={"size": (4500, 2500)}, lores={"size": (320, 240)}, display="lores")
        self.video_config = self.picam.create_video_configuration(display="lores")
        self.picam.configure(self.preview_config)

    def start(self ):
        self.picam.start()

    def stop(self):
        self.picam.stop()

    def preview(self):
        self.picam.start_preview(Preview.QT)

    def capture(self, type, name="test.jpg", wait=True):
        if type == 0:
            job = self.picam.capture_file(name, wait=wait)
        if type == 1:
            job = self.picam.capture_array(wait=wait)
        return wait


