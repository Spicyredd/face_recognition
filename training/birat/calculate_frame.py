import cv2
import face_recognition
from collections import Counter
import pickle
import numpy as np
from pathlib import Path
import time


DEFAULT_ENCODINGS_PATH = "output/encodings.pkl"

# Load encodings
def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

# Recognize face in video stream
def recognize_faces_in_video(encodings_location, model="hog", process_every_frame=1):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam+
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if frame_count % process_every_frame == 0:
            # Convert the image from BGR color to RGB color
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Recognize faces in the current frame
            for face_location, unknown_encoding in zip(face_locations, face_encodings):
                name = recognize_face(unknown_encoding, loaded_encodings)
                if not name:
                    name = "Unknown"
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

        # Calculate FPS
        frame_count += 1
        if time.time() - start_time >= 1:  # Every second
            fps = frame_count / (time.time() - start_time)
            print("FPS:", fps)
            frame_count = 0
            start_time = time.time()

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
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
        # Check if the highest vote count exceeds the threshold
        if votes[recognized_face] / sum(votes.values()) >= threshold:
            return recognized_face
    return None

# Run face recognition on live video
recognize_faces_in_video(DEFAULT_ENCODINGS_PATH, process_every_frame=30)  # Process every second frame
