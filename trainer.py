import cv2
import numpy as np
import os
import yaml

# Function to load images and labels
def load_images_and_labels(data_folder):
    images = []
    labels = []
    label_dict = {}  # Create a dictionary to map names to IDs
    current_id = 0  # Start with ID 0

    for label in os.listdir(data_folder):
        if not label.startswith('.'):  # Ignore hidden files or folders
            label_folder = os.path.join(data_folder, label)
            label_dict[label] = current_id  # Assign a unique ID to the label
            current_id += 1
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(label_dict[label])  # Append the corresponding ID

    return images, labels, label_dict



# Load images, labels, and label dictionary from the training folder
images, labels, label_dict = load_images_and_labels('training')

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the model
recognizer.train(images, np.array(labels))

# Save the model
recognizer.save('trained_model.yml')

correct_label = {value: key for key, value in label_dict.items()}
    
with open('labels.yml', 'w') as f:
    yaml.dump(correct_label, f)