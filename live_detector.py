import cv2
import face_recognition
from collections import Counter
import pickle
import numpy as np
from gradio_client import Client
import gradio_client
from PIL import Image

DEFAULT_ENCODINGS_PATH = "output/encodings.pkl"

# Load encodings
def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

# Recognize face in video stream
def recognize_faces_in_video(encodings_location, model="hog"):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Convert the image from BGR color to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
       
        # Recognize faces in the current frame
        for face_location, unknown_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location
            face = Image.fromarray(frame).crop((left, top, right, bottom))
            face.save('face.png')
            threshold = fake_detector('face.png')
            if threshold < 0.7:
                name = 'Fake: ' + recognize_face(unknown_encoding, loaded_encodings)
                if name == 'Fake: None':
                    name = "Fake: Unknown"
            else:
                name = recognize_face(unknown_encoding, loaded_encodings)
                if not name:
                    name = "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
            # Display the resulting image
            cv2.imshow('Video', frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Recognize face using loaded encodings
def recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:   
        print(votes)
        return votes.most_common(1)[0][0]

def fake_detector(image_location):
    client = Client("https://faceonlive-face-liveness-detection-sdk.hf.space/")
    result = client.predict(
            gradio_client.file(image_location),	# filepath  in 'parameter_4' Image component
            api_name="/face_liveness"
    )
    return result['data']['liveness_score']

# Run face recognition on live video
recognize_faces_in_video(DEFAULT_ENCODINGS_PATH)