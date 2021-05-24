from moviepy.editor import *
import cv2
from PIL import Image
import os
import time

from numpy.lib import utils
from utils import extract_face_cv2, extract_face_dlib
DATA_DIR = 'data/silent/'

WIDTH = 150;
HEIGHT = 150;

def data_for_rate(filepath, rate, debug):

    path = os.path.abspath(filepath)
    video_capture = cv2.VideoCapture(path)

    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT ) 
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)
    length = frame_count / frames_per_sec

    if (rate == None):
        rate = frames_per_sec

    #WORK WITH VIDEO
    d_time = 1 / rate
    timestamp = 0

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec) # optional
    success, frame = video_capture.read()

    images = []
    # c = 0
    cnt = 0
    # start_time = time.time()

    while success and timestamp < length:

        if (success == False):
            continue;

        res, face = extract_face_cv2(frame, debug)

        if (res):
            images.append(face)
            # c += 1
            cnt += 1

        # if (c == 100):
        #     print("--- 100 taken: %s ---" % (time.time() - start_time))
        #     start_time = time.time()
        #     c = 0

        # next frame
        timestamp = timestamp + d_time;
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec)
        success, frame = video_capture.read()

    print("total: " + str(cnt) + " images generated.")

    return images   

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail

def process_video_extract_data_using_rate(filepath, rate, debug):
    filename = path_leaf(filepath)

    print('generating data from ' + filename + '...')

    data = data_for_rate(filepath, rate, debug)
    print('done.')
    print('----------------------')

    return data
