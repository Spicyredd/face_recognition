import os
import cv2

data_folder = 'training'
label_dict = {}
current_id = 0
images = []
labels = []
for label in os.listdir(data_folder):
    if not label.startswith('.'):  # Ignore hidden files or folders
        label_folder = os.path.join(data_folder, label)
        label_dict[current_id] = label  # Assign a unique ID to the label
        current_id += 1
        
print(label_dict)
        
for label in os.listdir(data_folder):
    if not label.startswith('.'):  # Ignore hidden files or folders
        label_folder = os.path.join(data_folder, label)
        label_dict[label] = current_id  # Assign a unique ID to the label
        current_id += 1
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            if label_dict[current_id] == label:
                labels.append(current_id)  # Append the corresponding ID
label = {value: key for key, value in label_dict.items()}
print(label_dict)
print(labels)