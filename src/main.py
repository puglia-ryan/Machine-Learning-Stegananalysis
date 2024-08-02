import read_files

image_folder = 'data/images\/'
images, labels = read_files.read_and_resize_images(image_folder)

training_folder = 'data/train/train'
testing_folder = 'data/test/test'
val_folder = 'data/val/val'

training_imgs, training_labels = read_files.read_and_resize_images(training_folder)
testing_imgs, testing_labels = read_files.read_and_resize_images(testing_folder)
val_imgs, val_labels = read_files.read_and_resize_images(val_folder)


