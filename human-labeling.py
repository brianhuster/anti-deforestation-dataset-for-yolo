import os
import cv2
from ultralytics import YOLO

# Load your custom YOLO model
model = YOLO("/home/brianhuster/Downloads/yolov10l.pt")


# Define paths
dataset_path = "/media/brianhuster/D/Projects/Yolov9-deforestation-dataset/valid"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")

# Ensure the labels directory exists
os.makedirs(label_dir, exist_ok=True)


def get_predictions(model, img_path):
    return model(img_path)


def label_humans(image_path, label_path, img_width, img_height):
    # Run inference for human detection
    human_results = get_predictions(model, image_path)

    # Load existing labels if any
    existing_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            existing_labels = f.readlines()

    # Open the label file for writing
    with open(label_path, 'w') as f:
        # Write existing labels
        for label in existing_labels:
            print(repr(label))

            f.write(label)
            f.write("\n")

        # Write new human labels
        for result in human_results:
            for box in result.boxes.data:
                confidence = box[4].item()
                if confidence >= 0.5:
                    class_id = int(box[5].item())
                    if class_id == 0:  # Assuming class_id 0 is for humans in the pre-trained model
                        print(box)
                        x_min = box[0].item()
                        y_min = box[1].item()
                        x_max = box[2].item()
                        y_max = box[3].item()

                        # Convert to YOLO format
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        f.write(f"2 {x_center:.8f} {y_center:.8f} {
                                width:.8f} {height:.8f}\n")


# Iterate over all images in the dataset
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(
            label_dir, os.path.splitext(image_file)[0] + '.txt')

        # Read the image to get its dimensions
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        label_humans(image_path, label_path, img_width, img_height)

print("Labeling complete.")
