from PIL import Image
import os

def read_and_resize_images(image_folder, target_size=(1024, 1024)):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.png', '.jpeg', '.jpg'):
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size, Image.ANTIALIAS)
                images.append(img_resized)
    return images
