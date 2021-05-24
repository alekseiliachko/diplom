import numpy as np
import pandas
from keras.preprocessing import image_dataset_from_directory 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from matplotlib import pyplot
from skimage.transform import resize   # for resizing images
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D
import keras

X1 = np.load('npy/landmarks_talking.npy')
X2 = np.load('npy/landmarks_silent.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
Y = np.concatenate((Y1, Y2), axis=0).reshape(-1, 1)

print(X.shape)
print(Y.shape)

input_shape=(X.shape[1], X.shape[2])

print(input_shape)

model = Sequential()
model.add(InputLayer(input_shape=input_shape))

model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

history = model.fit(X, Y, epochs=15, validation_split=0.2, shuffle=True, batch_size=64)