import pickle
import sys
import time

import cv2
import face_recognition
import imutils
import numpy as np
from cv2.cv2 import VideoWriter
from imutils.video import FPS
from imutils.video import VideoStream


def draw_name_boxes(frame, boxes, names):
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)


def get_video_stream(frame_rate=32):
    if sys.platform.startswith('linux'):
        return VideoStream(usePiCamera=True, framerate=frame_rate)

    return VideoStream(src=0, framerate=frame_rate)


def process_frame(frame, detector, data):
    frame = imutils.resize(frame, width=500)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(data['encodings'], encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data['names'][best_match_index]

        # update the list of names
        names.append(name)

    draw_name_boxes(frame, boxes, names)

    return frame


if __name__ == '__main__':
    cascade = 'haarcascade_frontalface_default.xml'
    encodings = 'encodings.pickle'

    record = True

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodings, "rb").read())
    detector = cv2.CascadeClassifier(cascade)

    frame_rate = 32
    print("[INFO] starting video stream...")

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
        frame = process_frame(frame, detector, data)

        if record:
            if out is None:
                (h, w) = frame.shape[:2]
                out = VideoWriter("outpy.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (w, h))

            out.write(frame)

        # display the image to our screen
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    if record:
        out.release()
