import glob
import numpy as np
import cv2
image_pathes = glob.glob("calib_images\*.jpg")

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


imgpoints = []
objpoints = []

for fname in image_pathes:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
        imgpoints.append(corners)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
cv2.destroyAllWindows()
print(mtx)
choice = str(raw_input("fx und cx?"))
if(choice=="j"):
    fx = input("fx: ")
    cx = input("cx: ")
    mtx[0][0]=fx
    mtx[0][2]=cx
    print(mtx)

for i in range(0,len(objpoints)):
    test = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, None)
    img = cv2.imread(image_pathes[i])
    img = cv2.drawChessboardCorners(img, (9,6), test[0],int(ret))
    cv2.imshow('img',img)
    cv2.waitKey(500)



