import os
import shutil
import random

# Define paths
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Create directories for splits
output_dirs = ['train', 'val', 'test']
for split in output_dirs:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# List all image files
all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Shuffle and split
random.shuffle(all_images)
total = len(all_images)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

# Function to move files and labels


def move_files(image_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for image_name in image_list:
        # Move image
        shutil.move(os.path.join(src_image_dir, image_name),
                    os.path.join(dest_image_dir, image_name))

        # Move label file
        label_file = image_name.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(src_label_dir, label_file)):
            shutil.move(os.path.join(src_label_dir, label_file),
                        os.path.join(dest_label_dir, label_file))


# Move files to corresponding directories
move_files(train_images, images_dir, labels_dir, os.path.join(
    images_dir, 'train'), os.path.join(labels_dir, 'train'))
move_files(val_images, images_dir, labels_dir, os.path.join(
    images_dir, 'val'), os.path.join(labels_dir, 'val'))
move_files(test_images, images_dir, labels_dir, os.path.join(
    images_dir, 'test'), os.path.join(labels_dir, 'test'))
