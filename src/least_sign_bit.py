from PIL import Image
import numpy as np

def encode_lsb(image_path, message, output_path):
    #Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    #A flattened array allows for easier manipulation
    flat_img_array = img_array.flatten()

    #The message is encoded into binary
    message_in_bin = ''.join([format(ord(char), '80b') for char in message])
    
    for i in range(len(message_in_bin)):
        if message_in_bin[i] == '1':
            #The bitwise operator is used to determine if the last bit should change 
            flat_img_array[i] = flat_img_array[i] | 1
        else:
            flat_img_array[i] = flat_img_array[i] & ~1

    return 1