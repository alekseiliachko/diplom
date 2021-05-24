import numpy as np
import dlib
from PIL import Image
import cv2
import uuid

face_detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

WIDTH = 150
IMAGE_DEBUG_PATH = 'debug/image/'

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