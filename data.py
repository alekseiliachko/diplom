import numpy as np
from moviepy.editor import *
import cv2
import librosa.display
from PIL import Image

FILE_PATH = 'files/'
AUDIO_FOLDER = 'audio/'
A_DIR = "data/a/"
B_DIR = "data/b/"

AUDIO_NAME = 'audio.wav'

def normalizeLevel(level_map):
    codec_size = level_map.shape[0]
    size = level_map.shape[1];
    res = np.zeros(size)
    for i in range(0, size):
      sum = 0
      for codec in range(0, codec_size):
        sum += level_map[codec, i]
      res[i] = sum
    return res;

def getBorderValue(nparray):
    return ( sum(nparray) / nparray.size) / 4;

def processAudio(filename, rate):
    path = FILE_PATH + filename
    
    audioclip = AudioFileClip(path)
    audioclip.write_audiofile(AUDIO_FOLDER + AUDIO_NAME)

    sr = 22050
    n_fft = 1024
    hop_length = 16
    y, sr = librosa.load(AUDIO_FOLDER + AUDIO_NAME, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y)
    volume_level = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    normal_level = normalizeLevel(volume_level);
    length = int(librosa.get_duration(y))
    border = getBorderValue(normal_level);
    n_slices = int(round((length * rate),0))
    time_frame_length = int(volume_level.shape[1] / n_slices);
    return normal_level, border, length, time_frame_length

def processVideofile(filename, rate, normal_level, border, length, time_frame_length):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    path = FILE_PATH + filename
    
    path = os.path.abspath(path)

    video_capture = cv2.VideoCapture(path)

    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT ) 
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)

    #WORK WITH VIDEO
    d_time = 1 / rate
    timestamp = 0

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec) # optional
    success, frame = video_capture.read()

    format_ = '.jpg'
    flag = None
    image_path = None

    a_count = 0
    b_count = 0

    while success and timestamp < length:

        # go througn frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30))

        if (len(faces) != 0):
            x, y, w, h = faces[0]
            faceImage = frame[y:y+h, x:x+w]
            final = Image.fromarray(faceImage)

            if (normal_level[int(timestamp / d_time) * time_frame_length] > border):
                b_count += 1
                image_path = B_DIR + "b_" + str(b_count) + format_;
            else:
                a_count += 1
                image_path = A_DIR + "a_" + str(a_count) + format_;
        
        final.save(image_path)

        # next frame
        timestamp = timestamp + d_time;

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, timestamp * frames_per_sec)
        success, frame = video_capture.read()   


import sys

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    
    video__ = sys.argv[1]
    rate__ = int(sys.argv[2])

    normal_level_, border_, length_, time_frame_length_ = processAudio(video__, rate__)

    print("AUDIO DONE")

    processVideofile(video__, rate__,normal_level_, border_, length_, time_frame_length_)

    print("VIDEO DONE")
