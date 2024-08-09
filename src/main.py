import read_files
from PIL import Image
import numpy as np
import model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

training_folder = 'data/train/train'
training_folder_stego = 'data/train/train'
testing_folder = 'data/test/test'
val_folder = 'data/val/val'


training_imgs, training_labels = read_files.read_and_resize_images(training_folder)
testing_imgs, testing_labels = read_files.read_and_resize_images(testing_folder)
#val_imgs, val_labels = read_files.read_and_resize_images(val_folder)


model1 = model.build_model()
model1.fit(training_imgs, training_labels, epochs=10, batch_size=32, validation_data=(testing_imgs, testing_labels))
model1.save('my_model.h5')
 
