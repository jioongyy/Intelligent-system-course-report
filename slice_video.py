from imutils.video import VideoStream
import argparse
import imutils
import cv2
import os
import numpy as np

rotate_angle = 90
goal_size = 100
facial_recognize_side = 120
sample_frequency = 15

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where face cascade resides")
ap.add_argument("-i", "--input", required=True,
                help="path of video")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
vs = cv2.VideoCapture(args['input'])

total_frame = 0
count = 0

while (vs.isOpened()):
    ret, frame = vs.read()
    frame = imutils.rotate(frame, angle=rotate_angle)
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30)
    )
    # draw the rectangle in photo
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # crop the photo to the recognize area
    if (len(rects) != 0):
        side = max(rects[0][2], rects[0][3])
        frame = frame[rects[0][1]:rects[0][1] + side, rects[0][0]:rects[0][0] + side]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        if (side > facial_recognize_side) and (count % sample_frequency == 0):
            p = os.path.sep.join([args["output"], "{}.png".format(str(total_frame).zfill(5))])
            to_store = imutils.resize(frame, goal_size, goal_size)
            if np.array(to_store).shape == (goal_size,goal_size):
                print(to_store.shape)
                cv2.imwrite(p, to_store)
                total_frame += 1

    # show the output frame
    cv2.imshow("Frame", frame)

    # cv2.imshow("origin",orig)
    key = cv2.waitKey(1) & 0xFF
    #
    # if key == ord("k"):
    #     p = os.path.sep.join([args["output"], "{}.png".format(str(total_frame).zfill(5))])
    #     cv2.imwrite(p, orig)
    #     total_frame += 1
    # # if the 'q' key was pressed, break from the loop
    # elif key == ord("q"):
    #     break
