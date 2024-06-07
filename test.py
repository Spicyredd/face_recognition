import cv2

vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    cv2.imshow('video', frame)