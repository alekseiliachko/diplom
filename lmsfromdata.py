from utils import detect

def lms_for_peeks(images, debug):

    print("loaded: " + str(len(images)) + " faces.")
    lms = []

    i = 1
    for image in images:
        res, lm = detect(image, debug)
        if (res):
            lms.append(lm)
            i += 1

    print("total: " + str(len(lms)) + " landmarks.")
    return lms

def process_data_extract_lms(images, debug):

    print('generating lamdmarks for...')
    
    data = lms_for_peeks(images, debug)
    print('done.')
    print('----------------------')
    return data 

        