# from utils import extract_face_dlib
# from utils import extract_face_cv2
import cv2

_1_ = cv2.imread('test_files/1_.jpg',cv2.COLOR_RGB2BGR)
_2_ = cv2.imread('test_files/2_.jpg',cv2.COLOR_RGB2BGR)
_3_ = cv2.imread('test_files/3_.jpg',cv2.COLOR_RGB2BGR)

# _, face1 = extract_face_cv2(pil_image, True)
# _, face2 = extract_face_dlib(pil_image, True)

import matplotlib.pyplot as plt
plt.axis('off')

plt.subplot(1, 3, 1)
plt.imshow(_1_)

plt.subplot(1, 3, 2)
plt.imshow(_2_)

plt.subplot(1, 3, 3)
plt.imshow(_3_)

plt.show()


