import cv2
import os
import yaml
# Function to load images and labels

label_dict = {}
with open('labels.yml', 'r') as f:
    label_dict = yaml.safe_load(f)

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

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Read the model
recognizer.read('trained_model.yml')

# Function to draw rectangles and labels on detected faces
def draw_rectangle_with_label(image, rect, label):
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Capture video from webcam (replace 0 with camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        roi_gray = gray[y:y+h, x:x+w]

        # Recognize the face using the trained model
        label, confidence = recognizer.predict(roi_gray)

        # Get the name from the label_dict
        print(label)
        if label in label_dict:
            label_text = label_dict[label]
        else:
            label_text = 'Unknown'

        # Draw a rectangle and label around the face
        draw_rectangle_with_label(frame, (x, y, w, h), label_text)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()