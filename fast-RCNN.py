import tensorflow as tf
import data_prepare

from keras import models, layers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import Input

train_dir = data_prepare.train_dir
test_dir = data_prepare.test_dir
valid_dir = data_prepare.valid_dir

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=(224, 224))
valid_generator = valid_datagen.flow_from_directory(valid_dir, batch_size=16, target_size=(224, 224))

input_tensor = Input(shape=(224,224,3), dtype='float32', name='input')

pre_trained_vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
pre_trained_vgg.summary()

#additional_model = models.Sequential()
#additional_model.add(pre_trained_vgg)
#additional_model.add(layers.Flatten())
