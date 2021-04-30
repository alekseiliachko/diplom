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

directory = 'data/'

datagen = ImageDataGenerator(
      rotation_range=20,
      brightness_range=[0.2,1.0],
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      rescale=1./255)

generator = datagen.flow_from_directory(
    directory=directory,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
)

def generateImages_x_N(generator, batch_size, n):
  flags = []
  images = []

  for i in range(0, n):
    x, y = generator.next()
    m = x.shape[0]
    for j in range(0, m):
      image = x[j]
      flag = y[j]
      flags.append(flag)
      images.append(image)
  return np.array(images), np.array(flags)

X_, Y_ = generateImages_x_N(generator, 32, 200)

X = preprocess_input(X_, mode='tf')
Y = np_utils.to_categorical(Y_)

print(X.shape)
print(Y.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X, Y, steps_per_epoch=10, epochs=10, validation_split=0.2, shuffle=True, batch_size=20)