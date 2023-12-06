import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img1 = cv.imread('test3_1.png')
img2 = cv.imread('test3_2.png')
n, m, _ = np.shape(img1)
treeCover1 = 0
treeCover2 = 0
for i in range(0, n):
    for j in range(0, m):
        if img1[i, j, 0] < 30 and img1[i, j , 1] < 30 and img1[i, j, 2] < 30:
            treeCover1 += 1
        if img2[i, j, 0] < 30 and img2[i, j , 1] < 30 and img2[i, j, 2] < 30:
            treeCover2 += 1
if (n * m - treeCover1) * 4 < (n * m - treeCover2):
    print("Deforested detected")
else:
    print("Little Deforestation detected")