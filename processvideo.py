import numpy as np
from moviepy.editor import *
import cv2
from PIL import Image
import os
import sys
import random

# Load model
# model = smth()

FILE_PATH = 'files/'
WIDTH = 150;
HEIGHT = 150;

def process_stat(array, rate):
    s_time = 0
    f_time = 1 / rate
    for item in array:
        if (item):
            s_time += f_time;
    
    return s_time;

def process_video(filename, rate):
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

    res = []
    bln = False

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

            # do work
            # bln = model.process(final)
            bln = bool(random.getrandbits(1));
        else:
            bln = False
        
        res.append(bln)

        # next frame
        timestamp = timestamp + d_time;
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec)
        success, frame = video_capture.read()
    
    time_speaking = process_stat(res, rate);
    prc = time_speaking / length

    return time_speaking, prc

if __name__ == "__main__":
    
    filename = sys.argv[1]
    rate = int(sys.argv[2])

    time_, proc_ = process_video(filename, rate)
    print("processed video: " + filename)
    print("length of speech in the video: " + str(time_))
    print("part of the video: " + str(proc_))
