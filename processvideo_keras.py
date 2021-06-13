from moviepy.editor import *
import cv2
import os
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from utils import predict_frame, predict_frame_1d
from keras.models import load_model

clf = load_model('models/notmymodel')

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
        
        bln = predict_frame_1d(frame, clf=clf);
        
        res.append(bln)

        # next frame
        timestamp = timestamp + d_time;
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec)
        success, frame = video_capture.read()
    
    time_speaking = process_stat(res, rate);
    prc = time_speaking / length

    return time_speaking, prc

import time

if __name__ == "__main__":

    filename = sys.argv[1]
    rate = int(sys.argv[2])

    start_time = time.time()
    time_, proc_ = process_video(filename, rate, clf)
    print("processed video: " + filename)
    print("length of speech in the video: " + str(time_))
    print("part of the video: " + str(proc_))
    print("time taken: %s" % (time.time() - start_time))
