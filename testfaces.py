from utils import extract_face_dlib
from utils import extract_face_cv2
import cv2

pil_image = cv2.imread('test_files/image.jpg',cv2.COLOR_RGB2BGR)

_, face1 = extract_face_cv2(pil_image, True)
_, face2 = extract_face_dlib(pil_image, True)

import matplotlib.pyplot as plt
plt.axis('off')

plt.subplot(2, 2, 1)
plt.imshow(face1)

plt.subplot(2, 2, 2)
plt.imshow(face2)

plt.show()


