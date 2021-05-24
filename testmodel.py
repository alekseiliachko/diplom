import numpy as np
from keras.preprocessing import image_dataset_from_directory 
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from skimage.transform import resize   # for resizing images
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.layers import Activation, Dropout, Flatten, Dense
import keras

X1 = np.load('npy/talking_dataset.npy')
X2 = np.load('npy/silent_dataset.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
Y = np.concatenate((Y1, Y2), axis=0)

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