import read_files

training_folder_clean = 'data/train/train/clean\/'
training_folder_stego = 'data/train/train/stego\/'
testing_folder = 'data/test/test'
val_folder = 'data/val/val'


#training_imgs, training_labels = read_files.read_and_resize_images(training_folder)
#testing_imgs, testing_labels = read_files.read_and_resize_images(testing_folder)
#val_imgs, val_labels = read_files.read_and_resize_images(val_folder)

training_imgs_clean = read_files.read_2(training_folder_clean)
#training_imgs_stego = read_files.read_2(training_folder_stego)
print(training_folder_clean[1])
