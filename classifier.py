import os
import cv2
import math
import my_function
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

class_name = ['yangyi', 'yangjiakai']

yangyi_all_file = os.listdir("dataset/yangyi")
yangjiakai_all_file = os.listdir("dataset/yangjiakai")
yangyi_all_image = []
yangyi_all_label = []
yangjiakai_all_image = []
yangjiakai_all_label = []

# store all the photos and labels in array
for file in yangyi_all_file:
    open_image = np.array(Image.open("dataset/yangyi/" + file))
    if open_image.shape == (100, 100):
        yangyi_all_image.append(open_image)
        # yangyi_all_image.append(np.array(Image.open("dataset/yangyi/" + file)))
        yangyi_all_label.append(0)
    else:
        print("***********************************************************************************")
count = 0
for file in yangjiakai_all_file:
    open_image = np.array(Image.open("dataset/yangjiakai/" + file))
    if open_image.shape == (100, 100):
        yangjiakai_all_image.append(open_image)
        # yangjiakai_all_image.append(np.array(Image.open("dataset/yangjiakai/" + file)))
        yangjiakai_all_label.append(1)
    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

train_images = []
train_labels = []
test_images = []
test_labels = []

yangyi_chosen_list, yangyi_unchosen_list = \
    my_function.random_choose(len(yangyi_all_image), math.floor(len(yangyi_all_image) * 0.9))
for i in yangyi_chosen_list:
    train_images.append(yangyi_all_image[i])
    train_labels.append(yangyi_all_label[i])

for i in yangyi_unchosen_list:
    test_images.append(yangyi_all_image[i])
    test_labels.append(yangyi_all_label[i])

yangjiakai_chosen_list, yangjiakai_unchosen_list = \
    my_function.random_choose(len(yangjiakai_all_image), math.floor(len(yangjiakai_all_image) * 0.9))
for i in yangjiakai_chosen_list:
    train_images.append(yangjiakai_all_image[i])
    train_labels.append(yangjiakai_all_label[i])
for i in yangjiakai_unchosen_list:
    test_images.append(yangjiakai_all_image[i])
    test_labels.append(yangjiakai_all_label[i])

# shuffle the train images
zipped = list(zip(train_images, train_labels))
random.shuffle(zipped)
(train_images, train_labels) = zip(*zipped)

# plt.figure(figsize=(10, 10))
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_name[train_labels[i]])
# plt.show()

train_images = np.array(train_images)
test_images = np.array(test_images)

# scale the values to range 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("train.shape:{}".format(train_images.shape))
print("test.shape:{}".format(test_images.shape))

# model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
