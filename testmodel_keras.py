import numpy as np
from keras.preprocessing import image_dataset_from_directory 
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from skimage.transform import resize   # for resizing images
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, MaxPooling1D
import keras
from sklearn.model_selection import train_test_split

X1 = np.load('npy/talking_dataset.npy')
X2 = np.load('npy/silent_dataset.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
Y = np.concatenate((Y1, Y2), axis=0)

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)# in this our main data is splitted into train and test

print(X_train.shape[1::])

model = Sequential()

model.add(Conv1D(filters=80,kernel_size=16,strides=1,padding='valid',activation='elu',kernel_initializer='glorot_normal',input_shape=X_train.shape[1::]))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.5))

model.add(Conv1D(filters=40,kernel_size=9,strides=1,padding='same',activation='elu',kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), validation_batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("models/notmymodel")
print("Saved model to disk")