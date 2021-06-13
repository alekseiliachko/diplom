from datafrompeeks import process_video_extract_data_using_peeks
from datafromrate import process_video_extract_data_using_rate
from exctractpeeks import process_video_exctract_peeks
from lmsfromdata import process_data_extract_lms
import glob
import time
import numpy as np

TALKING_FOLDER = 'files/talking/*'
SILENT_FOLDER = 'files/silent/*'

def getFileNames(folder):
    filenames = glob.glob(folder)
    return filenames


TALKING_FILES = getFileNames(TALKING_FOLDER)
SILENT_FILES = getFileNames(SILENT_FOLDER)
# print(TALKING_FILES)

def load_silent_dataset():
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

def process_talking_videos():
    print('/////// PROCESSING TALKING VIDEOS... ///////')
    talking_dataset = []
    talking_total_time = 0
    start_time = time.time()
    for file in TALKING_FILES:
        __peeks = process_video_exctract_peeks(file, False)
        __data = process_video_extract_data_using_peeks(__peeks, file, False)
        __lms = process_data_extract_lms(__data, False)
        
        dtime = time.time() - start_time
        print("--- %s seconds ---" % (dtime))
        talking_dataset += __lms
        talking_total_time += dtime

    talking_dataset = np.array(talking_dataset)
    np.save('npy/talking_dataset', talking_dataset)

    print('/////// TOTAL: ' + str(talking_dataset.shape[0]))
    print('/////// DONE.                        ///////')

    return talking_total_time

def process_silent_videos():
    print('/////// PROCESSING SILENT  VIDEOS... ///////')
    silent_dataset = []
    silent_total_time = 0
    start_time = time.time()
    for file in SILENT_FILES:
        __data = process_video_extract_data_using_rate(file, None, False)
        __lms = process_data_extract_lms(__data, True)
        
        dtime = time.time() - start_time
        print("--- %s seconds ---" % (dtime))
        silent_dataset += __lms
        silent_total_time += dtime

    silent_dataset = np.array(silent_dataset)
    np.save('npy/silent_dataset', silent_dataset)
    
    print('/////// TOTAL: ' + str(silent_dataset.shape[0]))
    print('/////// DONE.                        ///////')

    return silent_total_time

t1 = process_silent_videos()
t2 = process_talking_videos()

print(' total time: ' + str(t1) + str(t2))