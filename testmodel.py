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
from keras.layers import Conv2D, MaxPooling2D
import keras

X1 = np.load('test/a/landmarks.npy')
X2 = np.load('test/b/landmarks.npy')
Y1 = np.zeros(X1.shape[0])
Y2 = np.ones(X2.shape[0])

X = np.concatenate((X1, X2), axis=0)
X = tf.expand_dims(X, axis=-1)
Y = np.concatenate((Y1, Y2), axis=0)

# XY = ?

print(X.shape)
print(Y.shape)

rows = X.shape[0]
cols = X.shape[1]
coord = X.shape[2]

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))

# model.summary()

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

input_shape=(rows, cols, coord)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

history = model.fit(X, Y, steps_per_epoch=10, epochs=10, validation_split=0.2, shuffle=True, batch_size=20)