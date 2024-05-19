import cv2
import face_recognition
import os
from collections import Counter
import pickle
import argparse

# --- Function Definitions --- 

def load_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

def recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    return votes.most_common(1)[0][0] if votes else None

def recognize_faces_in_video(encodings_location, model="hog", cpus=4):
    loaded_encodings = load_encodings(encodings_location)
    video_capture = cv2.VideoCapture(0)

    # Set the number of threads for OpenCV
    os.environ["OMP_NUM_THREADS"] = str(cpus)

    while True:
        ret, frame = video_capture.read()

        # Downscale image for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_location, unknown_encoding in zip(face_locations, face_encodings):
            # Scale back face locations
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            name = recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            print(name)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--encodings", type=str, default="output/encodings.pkl", help="Path to encodings file")
    parser.add_argument("--model", type=str, default="hog", choices=["hog", "cnn"], help="Face detection model (hog or cnn)")
    parser.add_argument("--cpus", type=int, default=4, help="Number of CPUs to use")
    args = parser.parse_args()

    recognize_faces_in_video(args.encodings, args.model, args.cpus)