from PIL import Image
import os
import numpy as np
from os import listdir


def read_and_resize_images(image_folder, target_size=(512, 512)):
    images = []
    labels = []
    i = 0
    for subdir, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png')):
                img_path = os.path.join(subdir, file)
                with Image.open(img_path) as img:
                    if img.mode == 'RGB':
                        img = img.convert('RGBA')
                    img_resized = img.resize(target_size, Image.ANTIALIAS)
                    images.append(np.asarray(img))
                    # This line may differ depending of the dataset's structure/labelling
                    labels.append(1 if 'stego' in subdir else 0)
        #images = np.array(images) / 255.0 #Normalisation
        
    return images, labels


