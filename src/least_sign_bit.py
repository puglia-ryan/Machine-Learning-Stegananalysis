from PIL import Image
import numpy as np

def encode_lsb(image_path, message, output_path):
        #Load the image
        img = Image.open(image_path)
        img_array = np.array(img)
    return