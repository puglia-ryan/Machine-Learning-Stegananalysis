from PIL import Image
import numpy as np

def encode_lsb(image_path, message, output_path):
    #Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    flat_img_array = img_array.flatten()

    #The message is encoded into binary
    message_in_bin = ''.join([format(ord(char), '80b') for char in message])
    
    return 1