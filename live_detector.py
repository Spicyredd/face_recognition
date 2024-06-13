import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from pytorch_train import cosine_similarity  # Assuming 'pytorch_train.py' contains your FaceNet class
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import warnings
warnings.simplefilter('ignore')


# ... (Your other code, including the `cosine_similarity` function, etc.
def recognize_face(model, frame, known_embeddings, known_labels):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Use the same size as during training
        transforms.ToTensor(),
    ])
    
    THRESHOLD = 0.6
    
    input_tensor = preprocess(frame).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()
        embedding = embedding.flatten()

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



with open('known_face_embeddings (3).pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

resnet = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)  # Move the model to the device

video_capture = cv2.VideoCapture(0)

start_time = 0
fps = 0

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=20,
    keep_all=True,
    select_largest=True
)

while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    end_time = time.time()
    fps = 1/ (end_time - start_time)
    start_time = time.time()
        
     # Perform face detection using MTCNN
    boxes, probs = mtcnn.detect(frame)
    face_frame = ''
    # Draw bounding boxes and landmarks (if available)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # left, top, right, bottom
            face_frame = frame[y1:y2, x1:x2]
            
            # Recognize faces in the frame
            predicted_label, max_similarity = recognize_face(resnet, face_frame, known_embeddings, known_labels)
            
        

            if predicted_label is not None:
                if max_similarity > 0.2:
                    # Display the predicted label and similarity on the frame
                    cv2.putText(frame, f'Name: {predicted_label}, Similarity: {max_similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                              
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video',frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()