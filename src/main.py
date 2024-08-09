import read_files
from PIL import Image
import numpy as np
import model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
import os
import pathlib
import matplotlib.pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split= 0.2,
    subset="training",
    seed=123,
    image_size=(512, 512))

val_ds = tf.keras.utils.image_dataset_from_directory(
  'data',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(512, 512))

data = data.map(lambda x, y: (x/255, y)) #Normalises the data from range 0-255 to 0-1
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

AUTOTUNE = tf.data.AUTOTUNE
data = data.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model1 = model.build_model()
model1.fit(data, validation_data=val_ds, epochs=3)
model1.save('my_model.h5')
"""
training_imgs, training_labels = read_files.read_and_resize_images(training_folder)
print("Done loading training images and labels")
testing_imgs, testing_labels = read_files.read_and_resize_images(testing_folder)
print("Done loading testing images and labels")
#val_imgs, val_labels = read_files.read_and_resize_images(val_folder)
training_imgs = np.asarray(training_imgs)
training_labels = np.asarray(training_labels)
testing_imgs = np.asarray(testing_imgs)
testing_labels = np.asarray(testing_labels)



model1 = model.build_model()
print("Model has been built")
model1.fit(training_imgs, training_labels, epochs=10, batch_size=1, validation_data=(testing_imgs, testing_labels))
model1.save('my_model.h5')
"""
