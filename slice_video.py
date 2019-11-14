import argparse
import imutils
import cv2
import os
import numpy as np
import my_function

# argument settings
rotate_angle = -90
goal_size = 100
facial_recognize_side = 100
sample_frequency = 15
# ******************

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
    # some videos direction is not correct cause opencv cannot recognize the face
    # so need to rotate the video
    frame = imutils.rotate(frame, angle=rotate_angle)
    frame = imutils.resize(frame, width=400)
    # opencv detect the face in the frame
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(facial_recognize_side, facial_recognize_side)
    )

    # crop the photo which including the recognize area
    if (len(rects) != 0):

        # print("rects: x:{} y:{} w:{} h:{}".format(rects[0][0],rects[0][1],rects[0][2],rects[0][3]))

        # opencv will detect all face appeare in the frame,and every face will store as a element of array
        # we need to find the biggest face area as the dataset from this array "rects"
        facial_area = my_function.primary_facial_area(rects)
        x, y, side = my_function.bigger_area(facial_area[0],facial_area[1],facial_area[2],facial_area[3],10)
        frame = frame[y:y + side, x:x + side]

        # print("modified: x:{} y:{} side:{}".format(x,y,side))

        # convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        if count % sample_frequency == 0:
            p = os.path.sep.join([args["output"], "{}.png".format(str(total_frame).zfill(5))])
            to_store = imutils.resize(frame, goal_size, goal_size)
            # the resize result probably is not conform to the goal_size, some deviation can be occured ocassinally
            # so we need to judge whether the resize result is standard or not
            if np.array(to_store).shape == (goal_size, goal_size):
                cv2.imwrite(p, to_store)
                total_frame += 1
