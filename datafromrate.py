import numpy as np
from moviepy.editor import *
import cv2
import librosa.display
from PIL import Image
import dlib 
import os
import sys
from imutils import face_utils
import uuid

FILE_PATH = 'files/'
DATA_DIR = 'data/silent/'

WIDTH = 150;
HEIGHT = 150;

def data_for_rate(filename, rate):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    path = FILE_PATH + filename
    path = os.path.abspath(path)
    video_capture = cv2.VideoCapture(path)

    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT ) 
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)
    length = frame_count / frames_per_sec

    #WORK WITH VIDEO
    d_time = 1 / rate
    timestamp = 0

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec) # optional
    success, frame = video_capture.read()

    while success and timestamp < length:

        if (success == False):
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
            final = final.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            image_path = DATA_DIR + str(uuid.uuid4()) + ".jpg"
            final.save(image_path)

        # next frame
        timestamp = timestamp + d_time;
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec)
        success, frame = video_capture.read()   



if __name__ == "__main__":
    
    filename = sys.argv[1]
    rate = int(sys.argv[2])

    data_for_rate(filename, rate)
