import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from model_train import cosine_similarity  # Assuming 'pytorch_train.py' contains your FaceNet class
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import warnings
warnings.simplefilter('ignore')
from queue import Queue, Empty
import threading

class FPSThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame_times = []
        self.fps_queue = Queue(maxsize=10)  # limit the queue size
        self.running = True

    def run(self):
        while self.running:
            # Sleep briefly to allow frame processing
            time.sleep(0.1)
            if len(self.frame_times) > 1:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
                if not self.fps_queue.full():
                    self.fps_queue.put(fps)

    def stop(self):
        self.running = False


# ... (Your other code, including the `cosine_similarity` function, etc.
def recognize_face(model, frame, known_embeddings, known_labels):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),  # Use the same size as during training
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

with open('vggface2.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

resnet = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)  # Move the model to the device

video_capture = cv2.VideoCapture(1)
start_time = 0
fps = 0

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20,
    keep_all=True,
    select_largest=True
)
ret, frame = video_capture.read()
width,height = frame.shape[:2]
print(f'Width:{width}, Height:{height}')


print('Video capture start.....')
while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()
    

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    width, height = frame.shape[:2]
    scale = 1
    mini_frame = cv2.resize(frame, (height//scale, width//scale))
    end_time = time.time()
    fps = 1/ (end_time - start_time)
    start_time = time.time()
        
    # Perform face detection using MTCNN
    boxes, probs = mtcnn.detect(mini_frame)
    face_frame = ''
    # Draw bounding boxes and landmarks (if available)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # left, top, right, bottom
            face_frame = mini_frame[y1:y2, x1:x2]
            x1, y1, x2, y2 = x1*scale , y1*scale , x2 *scale , y2 *scale
            
            # Recognize faces in the frame
            predicted_label, max_similarity = recognize_face(resnet, face_frame, known_embeddings, known_labels)

            if predicted_label is not None:
                if max_similarity > 0.6:
                    # Display the predicted label and similarity on the image
                    cv2.putText(frame, f'{predicted_label} {max_similarity:.2f}', (x1 + 10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Unknown Face", (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(y2,x2,x1,y2)
                              
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video',frame)
    cv2.imshow('Video_mini', mini_frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()



