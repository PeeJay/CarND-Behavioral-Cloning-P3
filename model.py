import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

path = "c:\\temp\\sim-data\\"
lines = []

with open (path + "driving_log.csv") as csvfile:

	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
	lines = shuffle(lines)

def generator(files, batch_size=32):
	num_samples = len(files)
	
	while 1:
		for n in range(0, num_samples, batch_size):
			images = []
			sAngles = []
			files = shuffle(files)
			for line in files[n:n+batch_size]:
				correction = 0.2
				#center image
				image = cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2RGB)
				images.append(image)
				images.append(cv2.flip(image, 1))
				sAngles.append(float(line[3]))
				sAngles.append(-float(line[3]))
				
				#left image
				image = cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB)
				images.append(image)
				images.append(cv2.flip(image, 1))
				sAngles.append(float(line[3]) + correction)
				sAngles.append(-(float(line[3]) + correction))

				#right image
				image = cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB)
				images.append(image)
				images.append(cv2.flip(image, 1))
				sAngles.append(float(line[3]) - correction)
				sAngles.append(-(float(line[3]) - correction))
				
			yield shuffle(np.array(images), np.array(sAngles))
	

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,3), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,3), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.summary()

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

history = model.fit_generator(
			train_generator, 
			validation_data=validation_generator,
			samples_per_epoch=len(train_samples)*6,
			nb_val_samples=len(validation_samples)*6,
			nb_epoch=15)
			
model.save("model.h5")

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
