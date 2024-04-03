from detector import recognize_faces
from cv2 import VideoCapture, imshow, imwrite, destroyWindow  

def camera_control():
    cam_port = 0
    cam = VideoCapture(cam_port) 
    result, image = cam.read() 

    if result:
        #display the camera image
        imshow("Camera", image)

        #save the image in local storage
        imwrite("data.png", image)

        #destroy the camera window after capturing the image
        # destroyWindow("Camera")

        recognize_faces('data.png')

    else: 
        print("No image detected. Please! try again")
camera_control()
# current_time = datetime.datetime.now()
# print("Current time:", current_time.strftime("%H:%M"))
# print(len(current_time.strftime("%H:%M")))