import cv2
import face_recognition
from collections import Counter
import pickle
from datetime import datetime

# Global variables
KNOWN_FACES = {}
LAST_RECOGNIZED = {}
DEFAULT_ENCODINGS_PATH = "output/encodings.pkl"
CREDENTIALS_FILE = "output/credentials.txt"


def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings


def recognize_faces_in_video(encodings_location, model="hog"):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    while True:
        ret, frame = video_capture.read()
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize frame for performance
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_location, unknown_encoding in zip(face_locations, face_encodings):
            name = recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            display_message(name)
            save_credentials(name)
            top, right, bottom, left = [coord*2 for coord in face_location]  # Scaling back to original frame size
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


def recognize_face(unknown_encoding, loaded_encodings, threshold=0.5):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=threshold
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        recognized_face = votes.most_common(1)[0][0]
        if votes[recognized_face] / sum(votes.values()) >= threshold:
            # Check if the face has not been recognized recently
            if recognized_face not in LAST_RECOGNIZED or (datetime.now() - LAST_RECOGNIZED[recognized_face]).seconds > 10:
                LAST_RECOGNIZED[recognized_face] = datetime.now()
                return recognized_face
    return None


def display_message(name):
    # Placeholder function to integrate with Tkinter for displaying messages
    pass


def save_credentials(name):
    # Save credentials to a text file
    with open(CREDENTIALS_FILE, 'a') as f:
        f.write(f"{name} - {datetime.now()}\n")


# Run face recognition on live video
recognize_faces_in_video(DEFAULT_ENCODINGS_PATH)
