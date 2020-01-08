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

labels = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':5,
           'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
           'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
           'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}


path_image = "JPEGImages_Sample"
path_annot = "Annotations_Sample"

train_size, test_size = 500, 250
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
    labels = np.zeros(shape=(sample_count, None))
    # Preprocess data
    generator = datagen.flow_from_directory(os.getcwd(), target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# train_features, train_labels = extract_features(path_image, train_size)  # Agree with our small dataset size
base_dir = os.getcwd()
train_x = [os.path.join(base_dir, i) for i in os.listdir(base_dir + "/train")]
valid_x = [os.path.join(base_dir, i) for i in os.listdir(base_dir + "/valid")]
test_x = [os.path.join(base_dir, i) for i in os.listdir(base_dir + "/test")]

#train_y = [labels[i] for i in range(len(train_x))]
#print(train_y)

def label_extract(data_set):
    mytype = data_set[0].split("/")[-1].split(".")[1]

    if mytype == "jpg":     # jpg타입 파일만 받습니다.
        for i in range(len(data_set)):
            num = data_set[i].split("/")[-1].split(".jpg")[0]   # 숫자만 추출합니다.
            tree = Et.parse(train_x + "/annot/{}".format_map(num) + ".xml") # xml 파일을 파싱해옵니다.
            root = tree.getroot()

            for member in root.findall('object'):
                name = member.find('name').text # 네이밍 단계




print(train_x[0].split("/")[-1].split(".jpg")[0])