To use the model follow these steps:

1. pip install -r requirements.txt(If you want to use you nvidia gpu make sure to install cuda version 12.1 and cudnn 9.x)
2. create folder with peoples name in both training and validation and put respective people's images in their folders
3. run preprocess.py to prepare data for model(new folders will be created faces_training and faces_validation)'
4. run model_train.py to train you new model
5. finally, run live_detector.py to run the model