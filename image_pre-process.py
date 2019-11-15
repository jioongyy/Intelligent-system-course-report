import os
import cv2
import argparse
import numpy as np
from PIL import Image
import my_function
import imutils
import uuid

facial_recognize_side = 30
goal_size = 100

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where face cascade resides")
# the folder store images in
ap.add_argument("-i", "--input", required=True,
                help="path of video")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

all_files = os.listdir(args["input"])
detector = cv2.CascadeClassifier(args["cascade"])

total_image = 0
for file in all_files:
    image = np.array(Image.open(args["input"] + file))
    rects = detector.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(facial_recognize_side, facial_recognize_side)
    )

    if len(rects) != 0:
        facial_area = my_function.primary_facial_area(rects)
        x, y, side = my_function.bigger_area(facial_area[0], facial_area[1], facial_area[2], facial_area[3], 29)
        image = image[y:y + side, x:x + side]

        # print("modified: x:{} y:{} side:{}".format(x,y,side))

        # convert the frame to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # use uuid to identify each frame
        p = os.path.sep.join([args["output"], "{}.png".format(uuid.uuid1())])
        to_store = imutils.resize(image, goal_size, goal_size)
        # the resize result probably is not conform to the goal_size,
        # some deviation can be occured ocassinally
        # so we need to judge whether the resize result is standard or not
        if np.array(to_store).shape == (goal_size, goal_size):
            cv2.imwrite(p, to_store)
            total_image += 1
        else:
            print("image {} failed to resize".format(file))

print("total image {}".format(total_image))