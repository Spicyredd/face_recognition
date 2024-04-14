import cv2 
import numpy as np
import face_recognition

Camera = cv2.VideoCapture(0)
_, frame = Camera.read()
frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
box = face_recognition.face_locations(frameRGB)

cx_ = (box[0][3] + box[0][1]) / 2
cy_ = (box[0][3] + box[0][1]) / 2
cx = cx_
cy = cy_
MIN_MOVE = 10

while True:
    _, frame = Camera.read()
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    box = face_recognition.face_locations(frameRGB)
    face_location = face_recognition.face_locations(frameRGB, model = "hog")
    top, right, bottom, left = face_location[0]
    if (box != []):
        cx = (box[0][3] + box[0][1]) / 2
        cy = (box[0][0] + box[0][2]) / 2
        cv2.rectangle(frame, (box[0][3], box[0][2]), (box[0][1],box[0][0]), (0,0,255) , 2 )
        
        if abs(cx - cx_) > abs(cy - cy_):
            if cx - cx_ > MIN_MOVE:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, 'Left', (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print('Left')
            elif cx - cx_ < -MIN_MOVE:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, 'Right', (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
                print('Right')
        else:
            if cy - cy_ > MIN_MOVE:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, 'Down', (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print('Down')
            elif cy - cy_ < -MIN_MOVE:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, 'Up', (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print('Up')
                
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(30)
    cx_ = cx
    cy_ = cy
    if key == ord('q'):
        break