import os
import cv2
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=20,
    keep_all=True,
    select_largest=True,
)


# Replace these with your actual paths
training_path = "training"
validation_path = "validation"

# Define the output folders
cropped_training_path = "faces_training"
cropped_validation_path = "faces_validation"

# Create output folders if they don't exist
os.makedirs(cropped_training_path, exist_ok=True)
os.makedirs(cropped_validation_path, exist_ok=True)

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_faces(folder_path, output_folder):
    """Crops faces from images in a given folder."""
    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            # Create subfolder for cropped images
            os.makedirs(os.path.join(output_folder, person_folder), exist_ok=True)
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Load the image
                    image = cv2.imread(image_path)
                    boxes, probs = mtcnn.detect(image)
                    try:
                      boxes = [[int(box[-1])]+[int(x) for x in box[:-1]] for box in boxes]
                      
                      for (top, right, bottom, left) in boxes:
                          # Crop the face
                          cropped_face = image[bottom:top, right:left]
                          # Save the cropped face
                          output_path = os.path.join(output_folder, person_folder, image_file)
                          cv2.imwrite(output_path, cropped_face)
                    except:
                      pass

# Process the training data
crop_faces(training_path, cropped_training_path)

# Process the validation data
crop_faces(validation_path, cropped_validation_path)

print("Cropping completed!")