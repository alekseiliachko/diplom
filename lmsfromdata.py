from utils import detect

DEBUG_PATH = 'debug/marked/'

def lms_for_peeks(images):

    print("loaded: " + str(len(images)) + " faces.")
    lms = []

    i = 1
    for image in images:
        res, lm = detect(image, False)
        if (res):
            lms.append(lm)
            i += 1

    print("total: " + str(len(lms)) + " landmarks.")
    return lms

def process_data_extract_lms(images):

    print('generating lamdmarks for...')
    
    data = lms_for_peeks(images)
    print('done.')
    print('----------------------')
    return data 

        