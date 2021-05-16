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

X = np.concatenate((X1, X2), axis=0)
# np.random.shuffle(X)   

Y = np.concatenate((Y1, Y2), axis=0)

print(X.shape)
print(Y.shape)

input_shape=(X.shape[1], X.shape[2])

model = Sequential()
model.add(Conv1D(32, kernel_size=3,
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# history = model.fit(X, Y, steps_per_epoch=10, epochs=10, validation_split=0.2, shuffle=True, batch_size=20)