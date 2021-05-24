from keras.models import load_model
import cv2
from utils import predict_frame_1d, predict_frame
import numpy as np
import pickle

X1 = np.load('npy/talking_dataset.npy')
X2 = np.load('npy/silent_dataset.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
Y = np.concatenate((Y1, Y2), axis=0)

# load model
model = load_model('models/mymodel')
clf = None
with open('models/trained', 'rb') as f:
    clf = pickle.load(f)

pil_image = cv2.imread('debug/image/0a1c51ac-8a83-4951-864b-04f26b933692.jpg',cv2.COLOR_RGB2BGR)

print(predict_frame(pil_image,clf))
print(predict_frame_1d(pil_image,model))