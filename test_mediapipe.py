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
warnings.simplefilter('ignore')

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
    scale = 1
    mini_frame = cv2.resize(frame, (height//scale, width//scale))
    mini_frame_copy = mp.Image(image_format = mp.ImageFormat.SRGB, data = mini_frame)
    end_time = time.time()
    fps = 1/ (end_time - start_time)
    start_time = time.time()
        
    
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
    
cv2.waitKey(0)
cv2.destroyAllWindows()