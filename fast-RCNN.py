import random
import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et

from PIL import Image
from keras.preprocessing import image
from pprint import pprint
from keras.applications import VGG16
from tensorflow_core.contrib.slim.python.slim.nets import vgg

labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


path_image = "JPEGImages_Sample/"
path_annot = "Annotations_Sample/"

train_size, test_size = 200, 100
print(train_size)
img_width, img_height = 224, 224


def show_pictures(path):
    random_img = random.choice(os.listdir(path))
    img_path = os.path.join(path, random_img)

    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
    plt.imshow(img_tensor)
    plt.show()

'''
for i in range(0, 2):
    show_pictures(path_image)
    show_pictures(path_image)
'''

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
conv_base.summary()

import os, shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        print(generator)
        print(inputs_batch)
        features_batch = conv_base.predict(inputs_batch)
        print(features_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        print(labels)
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(path_image, train_size)  # Agree with our small dataset size
print(train_features)
print(train_labels)