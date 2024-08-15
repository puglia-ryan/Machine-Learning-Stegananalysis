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
import least_sign_bit as lsb

lsb.encode_lsb("data/clean/00001.png", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "data/personal_stego/00001RickRoll.png")
#lsb.encode_all_img_in_folder("data/clean", "data/personal_stego")

data = tf.keras.utils.image_dataset_from_directory(
    'data/own_stego',
    image_size=(512, 512),
    batch_size=16)

data = data.map(lambda x, y: (x/255, y)) #Normalises the data from range 0-255 to range 0-1
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#The data is split into training, testing and validation sets
train = data.take(int(len(data)*0.7))
test = data.take(int(len(data)*0.2)+1)
val = data.take(int(len(data)*0.1)+1)


AUTOTUNE = tf.data.AUTOTUNE
data = data.cache().prefetch(buffer_size=AUTOTUNE)
#The number in the brackets denotes which model structure was used from the model.py file
checkpoint_path = "my_model3(1).h5"

#If the model with the given names exists, it is loaded in
if os.path.exists(checkpoint_path):
    model1 = load_model(checkpoint_path)
else:
    #else the model will be built
    model1 = model.build_model_2()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#The fit method allows the model to train and improve
#model1.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Following code block is to determine the accuracy of the model
true_labels = np.concatenate([y for x, y in test], axis=0)
predictions = model1.predict(test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = accuracy_score(true_labels, predicted_classes)
print(f'Accuracy: {accuracy * 100:2f}%')

#After training or prediciting, the model is once again saved
model1.save(checkpoint_path) 




