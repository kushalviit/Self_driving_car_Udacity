import csv
import cv2
import numpy as np
import os

samples=[]


with open('Data/driving_log.csv') as csvfile:
     reader=csv.reader(csvfile)
     for sample in reader:
         samples.append(sample)


#splitting training and validation data
from sklearn.model_selection import train_test_split
train_samples,validation_samples=train_test_split(samples,test_size=0.2)

# generator of batch data for validation and training
import sklearn
import random
def generator(samples,batch_size=32):
	num_samples=len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range (0,num_samples,batch_size):
			batch_samples=samples[offset:offset+batch_size]
			images=[]
			angles=[]
			for batch_sample in batch_samples:
				directory='Data/IMG/'
				img_center= cv2.imread(directory+batch_sample[0].split('/')[-1])
				img_left=cv2.imread(directory+batch_sample[1].split('/')[-1])
				img_right=cv2.imread(directory+batch_sample[2].split('/')[-1])
				steering_center=float(batch_sample[3])
				correction=0.3
				steering_left=steering_center+correction
				steering_right=steering_center-correction
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)
				images.append(np.fliplr(img_center))
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)
				angles.append((-(steering_center)))
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

#Calling the validation and training batch generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#Defining the neural network layer
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Activation,Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D

model=Sequential()
model.add(Lambda(lambda x:x/127.5-1.,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(60,3,3,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(72,3,3,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(84,3,3,subsample=(2,2),activation='relu'))
model.add(Dropout(0.4))
model.add(Convolution2D(96,2,2,activation='relu'))
model.add(Convolution2D(96,2,2,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#Compiling and feeding the generator data to train the network
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)*4), validation_data=validation_generator,nb_val_samples=(len(validation_samples)*4), nb_epoch=5)

#saving the model
model.save('model.h5')
