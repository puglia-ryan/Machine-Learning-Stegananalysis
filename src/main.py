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


#training_imgs, training_labels = read_files.read_and_resize_images(training_folder)
#testing_imgs, testing_labels = read_files.read_and_resize_images(testing_folder)
#val_imgs, val_labels = read_files.read_and_resize_images(val_folder)

#train1, train_label = read_files.read_and_resize_images("data\test_function\train")
#test1, test_label = read_files.read_and_resize_images("data\test_function\test")
model1 = model.build_model()
model1.summary()
train1 = np.random.rand(10, 512, 512, 4)
test1 = np.random.rand(5, 512, 512, 4)
train_label = np.random.randint(0, 2, 10)
test_label = np.random.randint(0, 2, 5)
model1.fit(train1, train_label, epochs=10, batch_size=32, validation_data=(test1, test_label))
model1.save('my_model.h5')
 
