import cv2
import numpy as np

img = cv2.imread("KITTI46_13.png", 1)


imgblau = img.copy()
imgblau[:,:,1]*=0
imgblau[:,:,2]*=0

imgrot = img.copy()
imgrot[:,:,0]*=0
imgrot[:,:,1]*=0

imggruen = img.copy()
imggruen[:,:,0]*=0
imggruen[:,:,2]*=0



b,g,r = cv2.split(img)
imgblau_split = cv2.merge((b,g*0,r*0))
imgrot_split = cv2.merge((b*0,g*0,r))
imggruen_split = cv2.merge((b*0,g,r*0))



cv2.imwrite("saves/blue.png",imgblau)
cv2.imwrite("saves/red.png",imgrot)
cv2.imwrite("saves/green.png",imggruen)
cv2.imwrite("saves/blue_split.png",imgblau_split)
cv2.imwrite("saves/red_split.png",imgrot_split)
cv2.imwrite("saves/green_split.png",imggruen_split)
