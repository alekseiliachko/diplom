import cv2
import numpy as np
import dlib 
from skimage.io import imread_collection
import os
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def loadImages(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def landmark(img, face):
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point

    # Look for the landmarks
    landmarks = predictor(image=img, box=face)
    l_data = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        l_data.append((x,y))
        cv2.circle(img=img, center=(int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)
    return img, l_data

def work(path, save_path):
    save_path_lm = save_path + "landmarks"
    images = loadImages(path)
    lms = []
    i = 1
    for image in images:
        faces = detector(image)
        if (len(faces) > 0):
            img, lm = landmark(image, faces[0])
            img = Image.fromarray(img)
            img.save(save_path + str(i) + ".jpeg")
            lms.append(lm)
            i += 1

    np.save(save_path_lm, np.array(lms))

work("data/a/", "test/a/")
work("data/b/", "test/b/")