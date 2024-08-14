import read_files
from PIL import Image
import numpy as np
import model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
import os
import pathlib
import matplotlib.pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(
    'data',
    image_size=(512, 512),
    batch_size=16)

data = data.map(lambda x, y: (x/255, y)) #Normalises the data from range 0-255 to 0-1
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

train = data.take(int(len(data)*0.7))
test = data.take(int(len(data)*0.2)+1)
val = data.take(int(len(data)*0.1)+1)


AUTOTUNE = tf.data.AUTOTUNE
data = data.cache().prefetch(buffer_size=AUTOTUNE)
#The number in the brackets denotes which model structure was used from the model.py file
checkpoint_path = "my_model2(2).h5"


if os.path.exists(checkpoint_path):
    model1 = load_model(checkpoint_path)
else:
    model1 = model.build_model_2()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#model1.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
true_labels = np.concatenate([y for x, y in test], axis=0)
predictions = model1.predict(test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = accuracy_score(true_labels, predicted_classes)
print(f'Accuracy: {accuracy * 100:2f}%')
model1.save(checkpoint_path) 

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
