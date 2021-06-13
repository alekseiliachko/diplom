import numpy as np
import dlib
from PIL import Image
import cv2
import uuid
from imutils import face_utils

face_detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

WIDTH = 150
IMAGE_DEBUG_PATH = 'debug/image/'
LMS_DEBUG_PATH = 'debug/marked/'

def lms_debug(image, lms): 
    for (x, y) in lms:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    cv2.imwrite(LMS_DEBUG_PATH + str(uuid.uuid4()) + ".jpg", image)



def extract_face_dlib(image, debug):

    detected_faces = face_detector(image, 1)
    if (len(detected_faces) == 0):
        return False, None
    face_rect = detected_faces[0]
    crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
    image_to_crop = Image.fromarray(image)
    cropped_image = cv2.cvtColor(np.array(image_to_crop.crop(crop_area).resize((WIDTH, WIDTH), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)
    
    if (debug):
        cv2.imwrite(IMAGE_DEBUG_PATH + str(uuid.uuid4()) + ".jpg", cropped_image)

    return True, cropped_image

def extract_face_cv2(image, debug):
    d = 10
    faces = faceCascade.detectMultiScale(
            image,
            # scaleFactor=2,
            minNeighbors=3,
            minSize=(100, 100))

    if (len(faces) == 0):
        return False, None
    x, y, w, h = faces[0]
    faceImage = image[y:y+h + d, x:x+w + d]    
    final = Image.fromarray(faceImage).resize((WIDTH, WIDTH), Image.ANTIALIAS)
    final = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    if (debug):
        cv2.imwrite(IMAGE_DEBUG_PATH + str(uuid.uuid4()) + ".jpg", final)
    return True, final

def landmark(img, face):
    landmarks = predictor(image=img, box=face)
    return face_utils.shape_to_np(landmarks).tolist()

def detect(image, debug):
    res, cut_face = extract_face_cv2(image, debug)
    if (not res):
        return False, None
    faces = face_detector(cut_face, 1)
    if (len(faces) > 0):
        lm = landmark(cut_face, faces[0])
        if (debug):
            lms_debug(cut_face, lm)
        return True, lm
    else:
        return False, None


def prepareLm(lm):
    return np.expand_dims(np.array(lm).flatten(), axis=0) / 150

def prepareLm1d(lm):
    a = np.expand_dims(np.array(lm), axis=0) / 150
    return a

def eval(arr):
    for elem in arr[0]:
        if elem > 0.5:
            return 1
    return 0

def predict_frame(frame, clf):
    success, lm = detect(frame, True)
        
    if (not success):
        return False

    processed = prepareLm(lm)

    prediction = clf.predict(processed)

    if (prediction[0] == 1):
        return True
    else:
        return False

def predict_frame_1d(frame, clf):
    success, lm = detect(frame, True)
        
    if (not success):
        return False

    processed = prepareLm1d(lm)

    prediction = eval(clf.predict(processed))
    
    if (prediction == 1):
        return True
    else:
        return False

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
#create your model
#then call the function on your model
