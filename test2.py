import cv2
from facenet_pytorch import MTCNN

image = cv2.imread('training\Anupama\IMG_20240605_094556.jpg')
# image2 = cv2.imread('training\Samyak\IMG_20240605_095147.jpg')

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=100,
    keep_all=True,
    select_largest=True
)

boxes, probs = mtcnn.detect(image)
boxes = [[int(box[-1])]+[int(x) for x in box[:-1]] for box in boxes]
top, right, bottom, left = boxes
# cv2.imshow('image',image[bottom:top, right:left])
cv2.waitKey(0)
print(boxes)