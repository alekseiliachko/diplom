import numpy as np
from moviepy.editor import *
import cv2
import librosa.display
from PIL import Image
import sys
import shutil
import pathlib
import os

FILE_FOLDER = 'files/'
AUDIO_FOLDER = 'audio/'
DATA_DIR = 'data/'
PEEK_PATH = 'npy/peeks.npy'

AUDIO_NAME = 'audio.wav'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def clear(path):
    path_ = os.path.abspath(path)
    shutil.rmtree(path_)
    os.makedirs(path_, exist_ok=True)

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

def images_for_peeks(filename, timespamps):
    path = FILE_FOLDER + filename
    path = os.path.abspath(path)
    video_capture = cv2.VideoCapture(path)
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)

    i = 0
    for timestamp in timespamps:
        frame_number = int_r(frames_per_sec * timestamp)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = video_capture.read()

        if (res == False):
            continue;

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30))
        
        if (len(faces) != 0):
            x, y, w, h = faces[0]
            faceImage = frame[y:y+h, x:x+w]
            final = Image.fromarray(faceImage)
            image_path = DATA_DIR + str(i) + ".jpg"
            final.save(image_path)
            i += 1

if __name__ == "__main__":
    filename = sys.argv[1]

    peeks = np.load(PEEK_PATH)
    print('Loaded ' + str(peeks.shape[0]) + ' peeks.')

    clear(DATA_DIR)

    images_for_peeks(filename, peeks)