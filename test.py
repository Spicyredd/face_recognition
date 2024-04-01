# import pandas as pd
import datetime
# data = pd.DataFrame({
#     'CMP 1234': ['10:15', '10:35', '09:40', '10:25', '08:55', '10:25', '08:45'],
#     }, index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
# print(data['CMP 1234']['Monday'])
# data.loc['Sunday', 'CMP 1234'] = '10:25'
# print(data.loc['Sunday', 'CMP 1234'])
# print(data['CMP 1234']['Monday'])
# print(data.head())
# data.head()

# current_time = datetime.datetime.now().strftime('%H:%M')
# print(current_time)

# def check_palin(word):
#   if len(word) == 0 or len(word) == 1:
#     return True

#   if word[0] == word[-1]:
#     return check_palin(word[1:-1])
  
#   print(word)
#   return False

# check_palin('Aibohphobia')


# class Done:
#     def get_val(self):
#         node = 1
#         self.node = 2
#         print(node)
#         print(self.node)

# done = Done()
# done.get_val()

from rembg import remove
from PIL import Image

# input_path = 'C:/Users/rbeej/Desktop/face_recognition/training/manjil/manjil0.jpg'
# output_path = 'C:/Users/rbeej/Desktop/face_recognition/training/manjil/manjil0.png'

# input = Image.open(input_path)
# input.save(output_path)

import glob
import os
from rembg import remove
from PIL import Image
import os
import sys
import pyuac
# pyuac.runAsAdmin()

# # Specify the directory path
# directory = 'C:/Users/rbeej/Desktop/face_recognition/training'

# # Use glob to get a list of all file paths
# file_paths = glob.glob(os.path.join(directory, '**/*'), recursive=True)
# pathhh = ''

# Iterate over the file paths and print only the file names
# for file_path in file_paths:
#     input_path = file_path
#     file_path = file_path.split('.')[0]
#     print(input_path)
#     input = Image.open(input_path)
#     input.save(file_path + '.png')
    
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
c = list(zip(a,b))
# for i,j in c:
#     print(i)
# print(c)

# from collections import Counter
# print(Counter(a))

import pickle
import face_recognition

loaded_encodings = ''
with open('output/encodings.pkl', 'rb') as f:
    loaded_encodings = pickle.load(f)
frame = Image.open('training/birat/birat2.jpg')
face_location = face_recognition.face_locations(frame, model = "hog")
face_encoding = face_recognition.face_encodings(frame, face_location)
for i in face_location:
    print(i)
boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], face_encoding)
print(boolean_matches)