import cv2
import face_recognition
from collections import Counter
import pickle
import time

DEFAULT_ENCODINGS_PATH = "output/encodings.pkl"

# Load encodings
def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

# Recognize face in video stream
def recognize_faces_in_video(encodings_location, model="hog", frame_resize_factor=0.25):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    while True:
        start_time = time.time()

        ret, frame = video_capture.read()

        # Resize frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resize_factor, fy=frame_resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for face_location, unknown_encoding in zip(face_locations, face_encodings):
            name = recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            top, right, bottom, left = face_location
            # Scale back up face locations since the frame we detected in was scaled to a smaller size
            top *= int(1/frame_resize_factor)
            right *= int(1/frame_resize_factor)
            bottom *= int(1/frame_resize_factor)
            left *= int(1/frame_resize_factor)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
            print(recognized_face)
            return recognized_face
    return None

recognize_faces_in_video(DEFAULT_ENCODINGS_PATH)
