import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

curr_path = os.path.dirname(__file__)
camry_path = os.path.join(curr_path, 'data/training/vehicle_images/camry_binary')
minivan_path = os.path.join(curr_path, 'data/training/vehicle_images/minivan_binary')
suv_path = os.path.join(curr_path, 'data/training/vehicle_images/suv_binary')

video = 'vid1.mp4'
vehicle_paths = [camry_path, minivan_path, suv_path]

training_images = []
labels = []

#Get image training data
for path in vehicle_paths:
    for image in os.listdir(path):
        image = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)  
        binary_image = cv2.resize(binary_image, (32,64))
        binary_image = binary_image.reshape(-1,1)

        training_images.append(binary_image)
        labels.append(1)

training_images = np.array(training_images)

# Creating random blobs
blobs = np.array([np.array([randint(0,1) for _ in range(2048)]) for _ in range(len(training_images))])
for _ in blobs:
    labels.append(0)

labels = np.array(labels)

training_images = np.squeeze(training_images, axis=2)

x_train,X_test,y_train, y_test = train_test_split(np.concatenate([training_images, blobs]), labels, test_size=0.3,random_state=102)

# Creating and training our neural network
model = Sequential()
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=x_train, y=y_train, epochs=15)

# Reading in the video feed
vid = cv2.VideoCapture(os.path.join(curr_path, f'data/videos/{video}'))

while(vid.isOpened()):
  # Capture frame-by-frame
  ret, frame = vid.read()
  if ret == True:
    # TODO: Find black blobs in video, check if the black blob is a car using NN
    pass 
 
  # Unable to get frame
  else: 
    break