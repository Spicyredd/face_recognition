# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import warnings
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from model_train import cosine_similarity  # Assuming 'pytorch_train.py' contains your FaceNet class
import cv2
from facenet_pytorch import InceptionResnetV1
import time
import warnings
import threading
from queue import Queue, Empty

warnings.simplefilter('ignore')

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

# Initialize FPS thread
fps_thread = FPSThread()
fps_thread.start()

prev_time = time.time()


# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

resnet = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)  # Move the model to the device

def get_face(
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  boxes = []
  for detection in detection_result.detections:    
    score = detection.categories[0].score
    bbox = detection.bounding_box
    start_point = bbox.origin_x - int((bbox.origin_x + bbox.width)*0.01), bbox.origin_y - int(bbox.origin_y*0.2) #left,top
    end_point = bbox.origin_x + bbox.width + int((bbox.origin_x + bbox.width)*0.01), bbox.origin_y + bbox.height #right, bottom
    x1, y1 = start_point
    x2, y2 = end_point
    boxes.append((x1, y1, x2, y2))
  return boxes
# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
vid = cv2.VideoCapture(1)
start_time = 0
fps = 0
_, frame = vid.read()

width, height = frame.shape[:2]
while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)
    scale = 1
    mini_frame = cv2.resize(frame, (height//scale, width//scale))
    mini_frame_copy = mp.Image(image_format = mp.ImageFormat.SRGB, data = mini_frame)
    # end_time = time.time()
    # fps = 1/ (end_time - start_time)
    # start_time = time.time()
    
    curr_time = time.time()
    time_diff = curr_time - prev_time
    prev_time = curr_time
        
        # Update frame times for FPS calculation
    fps_thread.frame_times.append(time_diff)
    if len(fps_thread.frame_times) > 10:  # Limit to the last 10 frame times
        fps_thread.frame_times.pop(0)
        
    try:
        fps = fps_thread.fps_queue.get(timeout=0.1)
    except Empty:
        fps = 0
    results = detector.detect(mini_frame_copy)
    boxes = get_face(detector.detect(mini_frame_copy))
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            # left, top, right, bottom
            face_frame = mini_frame[y1:y2, x1:x2]
            x1, y1, x2, y2 = [x*scale for x in box]
            # Recognize faces in the frame
            predicted_label, max_similarity = recognize_face(resnet, face_frame, known_embeddings, known_labels)

            if predicted_label is not None:
                if max_similarity > 0.6:
                    # Display the predicted label and similarity on the image
                    cv2.putText(frame, f'{predicted_label} {max_similarity:.2f}', (x1 + 10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Unknown Face", (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                              
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps_thread.stop()
fps_thread.join()
vid.release()
cv2.destroyAllWindows()