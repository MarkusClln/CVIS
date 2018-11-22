import cv2
import glob
import numpy as np

image_pathes = glob.glob("images\*.png")
img1 = cv2.imread(image_pathes[0])
img2 = cv2.imread(image_pathes[1])


blockSize = 10 # Gibt Fenstergroesse an (1-20)
min_disp = 10 # Gibt minimale Disparitaet an (0-10)
y=2
num_disp = 16*y # Gibt maxinale Disparitaet an

stereo = cv2.StereoSGBM_create(minDisparity = min_disp, numDisparities = num_disp, blockSize = blockSize)

disparity = stereo.compute(img1, img2).astype(np.float32)/16.0

print(img1)
print(disparity)
for n in range(0 ,len(disparity)):
    for m in range (0, len(disparity[n])):


[X, Y, Z, R, G, B]