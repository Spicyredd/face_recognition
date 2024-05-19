short description

This directory contains face images output by a face detector. There are 5105 client images and 7509 imposter images in total. Images from each subject are stored in a separate directory.

1. for evaluation
training set:
client   images: stored in the ClientFace directory and indexed in client_train_face.txt
imposter images: stored in the ImposterFace directory and indexed in imposter_train_face.txt

testing set:
client   images: stored in the ClientFace directory and indexed in client_test_face.txt
imposter images: stored in the ImposterFace directory and indexed in imposter_test_face.txt


2. The file name format of the indexed images: 

ID_galss_pos_session_picNo 

E.g. 0010_01_05_03_115.jpg
ID£º0001~0016 picture number
Glasses£º00~01
 	00£ºwith glasses
	01£ºno glasses
Pos£º   01~08 the location and light conditions of images
	01: up-down-rotate
        02: up-down-twist
        03: left-right-rotate
        04: left-right-twist
        05: close--window-open-lights
        07: open-window-open-lights
        08: open-widow-shut-lights
        08: still
Session£º01~03
picNo£ºpicture number

3. We also provide the coordinate of left eyes(x, y), right eyes(x, y), and the nose(x, y), all of which are detected by a facial 
feature detector. please see our eccv'10 paper for details about this.

E.g.\0010\0010_01_05_03_115.jpg 31.930000 31.930000 (left eyes) 67.980000 33.990000 (right eyes) 49.440000 56.650000 (nose)

