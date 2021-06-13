from exctractpeeks import process_video_exctract_peeks
import librosa
import matplotlib.pyplot as plt
from exctractpeeks import butter_highpass
from datafrompeeks import process_video_extract_data_using_peeks
from lmsfromdata import process_data_extract_lms
import cv2
import numpy as np

filepath = "files/d.mp4"
audiopath = "audio/d.mp4_audio.wav"

# from utils import detect, extract_face_cv2
# y, sr = librosa.load(audiopath)
# x_f=butter_highpass(y,30000, sr, order=5)
# y = librosa.onset.onset_strength(x_f, sr=sr)
# fig = plt.plot()
# librosa.display.waveplot(y, sr=sr)
# plt.show()

# peeks = process_video_exctract_peeks(filepath, False)
__data = process_video_extract_data_using_rate(filepath, 3, True)

# data = process_video_extract_data_using_peeks(peeks, filepath, False)
# lms = process_data_extract_lms(data, True)

# image = cv2.imread('test_files/image.jpg',cv2.COLOR_RGB2BGR)

# # extract_face_cv2(image, True)
# _, lm = detect(image, False)
# lm = np.array(lm) / 150

# fig = plt.plot()

# x = lm[:,0]
# y = lm[:,1]
# plt.scatter(x, y, s = 5)

# plt.show();
# plt.imshow(face1)





