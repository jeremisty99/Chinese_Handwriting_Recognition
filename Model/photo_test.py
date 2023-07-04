import matplotlib.pyplot as plt
from cv2 import cv2

from test import predict_multiple

img = cv2.imread('2.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img = clahe.apply(img)
# retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
# plt.imshow(img)
# plt.show()
print(predict_multiple(img))

