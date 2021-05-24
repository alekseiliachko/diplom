from datafrompeeks import process_video_extract_data_using_peeks
from datafromrate import process_video_extract_data_using_rate
from exctractpeeks import process_video_exctract_peeks
from lmsfromdata import process_data_extract_lms
import glob
import time

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
    start_time = time.time()
    for file in TALKING_FILES:
        __peeks = process_video_exctract_peeks(file, False)
        __data = process_video_extract_data_using_peeks(__peeks, file, False)
        process_data_extract_lms(__data, 'talking', False)
    print("--- %s seconds ---" % (time.time() - start_time))

def process_silent_videos():
    start_time = time.time()
    for file in SILENT_FILES:
        __data = process_video_extract_data_using_rate(file, None, False)
        process_data_extract_lms(__data, 'silent', True)
    print("--- %s seconds ---" % (time.time() - start_time))

process_silent_videos()