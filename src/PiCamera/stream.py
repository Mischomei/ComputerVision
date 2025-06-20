from picamera2 import Picamera2, Preview


def pi_stream(cam_num):
    picam = Picamera2()
    preview_config = picam.create_preview_configuration()
    still_config = picam.create_still_configuration(lores={"size": (320, 240)}, display="lores")
    video_config = picam.create_video_configuration()
    picam.configure(preview_config)
    picam.start_preview(Preview.QT)
    picam.start()



