import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model_1():
    num_classes = 2
    model1 = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model1.compile(
            optimizer=Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
            )
    return model1



def build_model_1():
    num_classes = 2
   

def build_model_2(input_shape=(512, 512, 3)):
    num_classes = 2
    model2 = tf.keras.models.Sequential()

    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Flatten())
    model2.add(Dense(512, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(2, activation='softmax'))

    model2.compile(
                optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model2

