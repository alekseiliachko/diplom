import numpy as np
from moviepy.editor import *
import cv2
from PIL import Image
import os
import sys
import random
import pickle
from utils import predict_frame

clf = None
with open('models/trained', 'rb') as f:
    clf = pickle.load(f)

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

def process_video(filename, rate, clf):

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
        
        bln = predict_frame(frame, clf=clf);
        
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

    time_, proc_ = process_video(filename, rate, clf)
    print("processed video: " + filename)
    print("length of speech in the video: " + str(time_))
    print("part of the video: " + str(proc_))
