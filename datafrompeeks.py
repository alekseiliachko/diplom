import numpy as np
from moviepy.editor import *
import cv2
from PIL import Image
import shutil
import os
from utils import extract_face_cv2
from utils import extract_face_dlib

AUDIO_FOLDER = 'audio/'
DATA_DIR = 'data/talking/'

NPY_PATH = 'npy/'
NPY_POSTF = '_peeks.npy'

AUDIO_NAME = 'audio.wav'

def clear(path):
    path_ = os.path.abspath(path)
    shutil.rmtree(path_)
    os.makedirs(path_, exist_ok=True)

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

def images_for_peeks(filepath, timespamps, debug):
    
    path = os.path.abspath(filepath)
    video_capture = cv2.VideoCapture(path)
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)

    cnt = 0
    images = []

    for timestamp in timespamps:
        frame_number = int_r(frames_per_sec * timestamp)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = video_capture.read()

        if (not res):
            continue;

        res, face = extract_face_dlib(frame, debug)

        if (res):
            images.append(face)
            cnt += 1

    print("total: " + str(cnt) + " faces.")

    return images

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail

def process_video_extract_data_using_peeks(peeks, filepath, debug):
    filename = path_leaf(filepath)
    print('generating data from peeks for ' + filename + '...')

    print('loaded: ' + str(peeks.shape[0]) + ' peeks.')

    data = images_for_peeks(filepath, peeks, debug)

    print('done.')
    print('----------------------')

    return data