import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import torchvision.transforms as transforms
import torch
from model_train import cosine_similarity  # Assuming 'pytorch_train.py' contains your FaceNet class
from facenet_pytorch import InceptionResnetV1
import pickle
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
    
mp_face_detection = mp.solutions.face_detection
cap = cv.VideoCapture(1)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)  # Move the model to the device

with open('vggface2.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Extract known embeddings and labels
known_embeddings = [embedding for embedding, _ in known_face_embeddings]
known_labels = [label for _, label in known_face_embeddings]

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
    frame_counter = 0
    fonts = cv.FONT_HERSHEY_PLAIN
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        face_frame = ''
        if ret is False:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)
                x, y, w, h = face_react
                face_frame = frame[y:y+h, x:x+w]
                predicted_label, max_similarity = recognize_face(resnet, face_frame, known_embeddings, known_labels)
                if predicted_label is not None:
                    if max_similarity > 0.6:
                        # Display the predicted label and similarity on the image
                        cv.putText(frame, f'{predicted_label} {max_similarity:.2f}', (x + h , y + h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        cv.putText(frame, "Unknown Face", (x + h , y + h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.rectangle(frame, face_react, color=(255, 255, 255), thickness=2)
                print(len(face_react))
        fps = frame_counter / (time.time() - start_time)
        cv.putText(frame,f"FPS: {fps:.2f}",(30, 30),cv.FONT_HERSHEY_DUPLEX,0.7,(0, 255, 255),2,)
        cv.imshow("frame", frame)
        # cv.imshow('frame', cropped_face)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows() 