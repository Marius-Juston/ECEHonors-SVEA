import socket
import sys
import time

from imutils.video import FPS
from imutils.video import VideoStream

import imagezmq.imagezmq


def get_video_stream(frame_rate=32):
    if sys.platform.startswith('linux'):
        return VideoStream(usePiCamera=True, framerate=frame_rate)

    return VideoStream(src=0, framerate=frame_rate)


if __name__ == '__main__':

    server_ip = 'localhost'
    frame_rate = 32
    print("[INFO] starting video stream...")

    sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(server_ip))
    rpiName = socket.gethostname()

    vs = get_video_stream(frame_rate).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    out = None

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()

        sender.send_image(rpiName, frame)
