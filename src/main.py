import read_files

image_folder = 'data/images\/'
images, labels = read_files.read_and_resize_images(image_folder)

# Split the images into training and validation data sets 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
