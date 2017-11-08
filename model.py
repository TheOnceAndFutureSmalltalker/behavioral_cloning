import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D


# read lines from log file
lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))


# load images and steering angles as indicated
# by log file entries
images = []
measurements = []
for line in lines:
    source_path = line[0]  
    filename = source_path.split('\\')[-1]
    current_path = './data/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print(len(images))
print(len(measurements))
X_train = np.array(images)
y_train = np.array(measurements)
print(X_train[0].shape)


# define NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

          
# train the model and save
model.compile(loss='mse', optimizer='adam')
from keras import backend as K
with K.get_session():
    model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=3)
    model.save('model.h5')
print('Done')
