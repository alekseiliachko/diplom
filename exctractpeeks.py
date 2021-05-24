import librosa
import librosa.display
import numpy as np
import datetime
import sys
from moviepy.editor import AudioFileClip
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

FILE_FOLDER = 'files/'

AUDIO_FOLDER = 'audio/'
AUDIO_POSTF = '_audio.wav'

NPY_PATH = 'npy/'
NPY_POSTF = '_peeks.npy'

def butter_highpass(data,cutoff, fs, order=5):
   """
   Design a highpass filter.
   Args:
   - cutoff (float) : the cutoff frequency of the filter.
   - fs     (float) : the sampling rate.
   - order    (int) : order of the filter, by default defined to 5.
   """
   # calculate the Nyquist frequency
   nyq = 2 * fs
   # design filter
   high = cutoff / nyq
   b, a = butter(order, high, btype='high', analog=False)
   # returns the filter coefficients: numerator and denominator
   y = filtfilt(b, a, data)
   return y

if __name__ == "__main__":    

    do_print = False

    if (len(sys.argv) > 2):
        do_print = bool(sys.argv[2])

    filename = sys.argv[1]

    file_path = FILE_FOLDER + filename
    npy_path = NPY_PATH + filename + NPY_POSTF
    audio_path = AUDIO_FOLDER + filename + AUDIO_POSTF
    
    audioclip = AudioFileClip(file_path)
    audioclip.write_audiofile(audio_path)

    audio, sr = librosa.load(audio_path)

#   FILTER
    x_f=butter_highpass(audio,30000, sr, order=5)

    o_env = librosa.onset.onset_strength(x_f, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.util.peak_pick(o_env, 2, 3, 3, 5, 0.3, 4)

    peeks = np.array(librosa.frames_to_time(onset_frames, sr=sr))
    
    if (do_print):
        print(peeks)

        plt.plot(times, o_env, label='Onset strength')
        plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
        linestyle='--', label='Onsets')

        plt.show()
    
    print("total: " + str(peeks.shape[0]) + " peeks found.")
    np.save(npy_path, peeks)