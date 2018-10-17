import cv2
import numpy as np

img = cv2.imread("KITTI46_13.png", 1)

print(img.shape)
imgAuto = img[150:300,780:1070,:]

img[200:350,40:330,:]=imgAuto

#cv2.imshow("Bildanzeige",img)
#cv2.waitKey(0)

cv2.imwrite("saves/Aufgabe2.png",img)