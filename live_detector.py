import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from pytorch_train import FaceNet, cosine_similarity
import cv2
from yolov5facedetector.face_detector import YoloDetector
from facenet_pytorch import MTCNN
import time


# ... (Your other code, including the `FaceNet` class, `cosine_similarity` function, etc.)

def recognize_face(model, frame, known_embeddings, known_labels):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Use the same size as during training
        transforms.ToTensor(),
    ])
    
    input_tensor = preprocess(frame).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()

    # Calculate cosine similarity with each known embedding
    similarities = []
    for known_embedding in known_embeddings:
        similarity = cosine_similarity(embedding, known_embedding.T)
        similarities.append(similarity)

    if similarities:
        max_similarity = np.max(similarities)
        predicted_label = known_labels[np.argmax(similarities)]
        return predicted_label, max_similarity
    else:
        return None, 0.0

# Load the model
model = FaceNet(embedding_size=128)  # Use the same embedding size as during training
model.load_state_dict(torch.load('facenet_model.pth'))
model.eval()  # Set to evaluation mode

# Load the known embeddings
with open('known_face_embeddings.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

video_capture = cv2.VideoCapture(0)


start_time = 0
fps = 0



mtcnn = MTCNN(
    image_size=224,
    margin = 20,
    min_face_size = 20,
    keep_all=True,
    select_largest=True
)

while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Recognize faces in the frame
    predicted_label, max_similarity = recognize_face(model, frame, known_embeddings, known_labels)

    if predicted_label is not None:
        # Display the predicted label and similarity on the frame
        cv2.putText(frame, f'Name: {predicted_label}, Similarity: {max_similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Unknown Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    end_time = time.time()
    fps = 1/ (end_time - start_time)
    start_time = time.time()
        
     # Perform face detection using MTCNN
    boxes, probs, points = mtcnn.detect(frame, landmarks=True)

    # Draw bounding boxes and landmarks (if available)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks (if available)
            if points is not None:
                for point in points[i]:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

# ... (Load the model and embeddings as shown above)

# image_path = 'rishav7.jpg'
# predicted_label, max_similarity = recognize_face(model, image_path, known_embeddings, known_labels)

# if predicted_label is not None:
#     print(f'Predicted label: {predicted_label}, Similarity: {max_similarity:.4f}')
# else:
#     print("No matching face found.")