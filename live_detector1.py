import cv2
import face_recognition
from collections import Counter
import pickle
import threading
import queue

DEFAULT_ENCODINGS_PATH = "output/encodings.pkl"

# Load encodings
def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

# Recognize face in video stream
def recognize_faces_in_video(encodings_location, model="hog"):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    # Create frame buffer and display buffer
    frame_buffer = queue.Queue(maxsize=10)  # Increased buffer size
    display_buffer = queue.Queue(maxsize=10)  # Queue for display

    # Function to capture frames from the camera
    def capture_frames():
        while not exit_flag.is_set():
            ret, frame = video_capture.read()
            if ret:
                frame_buffer.put(frame)

    # Function to recognize faces in frames
    def process_frames():
        while not exit_flag.is_set():
            if not frame_buffer.empty():
                frame = frame_buffer.get()
                recognize_faces(frame, loaded_encodings, model, display_buffer)

    # Function to display frames
    def display_frames():
        while not exit_flag.is_set():
            if not display_buffer.empty():
                frame = display_buffer.get()
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag.set()
                    break

    # Start capture, processing, and display threads
    exit_flag = threading.Event()
    threading.Thread(target=capture_frames).start()
    threading.Thread(target=process_frames).start()
    threading.Thread(target=display_frames).start()  # Display thread

    # Wait for exit flag to be set
    exit_flag.wait()

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Recognize faces in a single frame
def recognize_faces(frame, loaded_encodings, model="hog", display_buffer=None):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_location, unknown_encoding in zip(face_locations, face_encodings):
        name = recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if display_buffer:
        display_buffer.put(frame)  # Put frame into the display buffer

# Recognize a single face encoding
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
            return recognized_face
    return None

# Run face recognition on live video
recognize_faces_in_video(DEFAULT_ENCODINGS_PATH)
