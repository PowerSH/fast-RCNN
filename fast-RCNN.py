import data_prepare
import myselectivesearch
import preparing

import os
import tensorflow.compat.v1 as tf
import ROI_pooling
import numpy as np
import cv2
import xml.etree.ElementTree as Et
import pandas as pd

from keras import models, layers
from keras.applications import VGG16
from keras import Input

# associate compat.v1
tf.disable_v2_behavior()

train_dir = data_prepare.train_dir
test_dir = data_prepare.test_dir
valid_dir = data_prepare.valid_dir

train_image_dir = data_prepare.train_image_dir
test_image_dir = data_prepare.test_image_dir
valid_image_dir = data_prepare.valid_image_dir

train_annot_dir = data_prepare.train_annot_dir
test_annot_dir = data_prepare.test_annot_dir
valid_annot_dir = data_prepare.valid_annot_dir

input_tensor = Input(shape=(224,224,3), dtype='float32', name='input')

pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
#pre_trained_vgg.summary()
'''
# ROI part -start
batch_size = 1
img_height = 224
img_width = 224
n_channels = 1
n_rois = 2
pooled_height = 7
pooled_width = 7

feature_maps_shape = (batch_size, img_height, img_width, n_channels)
feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
feature_maps_np = np.ones(feature_maps_tf.shape, dtype='float32')
feature_maps_np[0, img_height-1, img_width-3, 0] = 50
print(f"feature_maps_np.shape = {feature_maps_np.shape}")

roiss_tf = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
roiss_np = np.asarray([[[0.5,0.2,0.7,0.4], [0.0,0.0,1.0,1.0]]], dtype='float32')
print(f"roiss_np.shape = {roiss_np.shape}")

roi_layer = ROI_pooling.ROIPoolingLayer(pooled_height, pooled_width)
pooled_features = roi_layer([feature_maps_tf, roiss_tf])
print(f"output shape of layer call = {pooled_features.shape}")

with tf.Session() as session:
    result = session.run(pooled_features,
                         feed_dict={feature_maps_tf: feature_maps_np,
                                    roiss_tf: roiss_np})

print(f"result.shape = {result.shape}")
print(f"first  roi embedding=\n{result[0, 0, :, :, 0]}")
print(f"second roi embedding=\n{result[0, 1, :, :, 0]}")
# ROI Part -end
'''
X = []
Y = []

X = preparing.dataprepare(train_image_dir)
Y = preparing.labeling(train_annot_dir)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

#print(preparing.labels.keys())
#temp = pd.DataFrame(Y, columns=preparing.labels.keys())
#print(temp)


additional_model = models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(layers.Flatten())

additional_model.add(layers.Dense(4096, activation='relu', name='fc1'))
additional_model.add(layers.Dense(2048, activation='relu', name='fc2'))
additional_model.add(layers.Dense(20, activation='softmax', name='classifier'))

additional_model.compile(optimizer='adam', loss='mean_squared_error')

hist = additional_model.fit(X, batch_size=128, epochs=5)

print("additional_model done")
