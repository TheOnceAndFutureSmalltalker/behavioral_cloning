import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))
images = []
measurements = []
for line in lines:
    source_path = line[0]
    speed = float(line[6])
    # ignore low speeds from startup as transient
    # data since car is pointing into wall for several frames!
    #if(speed > 10):  
    filename = source_path.split('\\')[-1]
    #current_path = './data/IMG/' + filename
    current_path = './data/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print(len(images))
print(len(measurements))

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
print('added flipped images')

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train[0].shape)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))

# LeNet/AlexNet
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

# NVIDIA Architecture
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

          

model.compile(loss='mse', optimizer='adam')
from keras import backend as K
with K.get_session():
    model.fit(X_train, y_train, validation_split=0.02, shuffle=True, nb_epoch=3)
    model.save('model.h5')
    
#import gc
#gc.collect()
print('Done')
