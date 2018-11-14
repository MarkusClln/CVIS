from siftdetector import detect_keypoints
import glob
import cv2
import numpy as np
import pickle

image_pathes = glob.glob("images\*.png")
def set_keypoints():
    [detected_keypoints1, descriptors1] = detect_keypoints(image_pathes[0], 5)
    [detected_keypoints2, descriptors2] = detect_keypoints(image_pathes[1], 5)
    [detected_keypoints3, descriptors3] = detect_keypoints(image_pathes[2], 5)
    [detected_keypoints4, descriptors4] = detect_keypoints(image_pathes[3], 5)
    with open('keypoints.pkl', 'w') as f:
        pickle.dump([detected_keypoints1, descriptors1,detected_keypoints2,descriptors2,detected_keypoints3,descriptors3,detected_keypoints4,descriptors4], f)



with open('keypoints.pkl') as f:
    detected_keypoints1, descriptors1, detected_keypoints2, descriptors2, detected_keypoints3, descriptors3, detected_keypoints4, descriptors4 = pickle.load(f)

def to_cv2_kplist(kp):
    return list(map(to_cv2_kp, kp))

def to_cv2_kp(kp):
    return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3]/np.pi*180)

def to_cv2_di(di):
    return np.asarray(di, np.float32)

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def matches_1():
    descriptors_cv2_1 = to_cv2_di(descriptors1)
    descriptors_cv2_2 = to_cv2_di(descriptors2)

    keypoints_cv2_1 = to_cv2_kplist(detected_keypoints1)
    keypoints_cv2_2 = to_cv2_kplist(detected_keypoints2)


    bf = cv2.BFMatcher()

    img1 = cv2.imread(image_pathes[0])
    img2 = cv2.imread(image_pathes[1])

    matches = bf.knnMatch(descriptors_cv2_1,descriptors_cv2_2, k=2)
    good = []
    pts1 = []
    pts2 = []
    theshold_matching = 0.7
    for m,n in matches:
        if m.distance < theshold_matching*n.distance:
            good.append([m])
            pts1.append(keypoints_cv2_1[m.queryIdx].pt)
            pts2.append(keypoints_cv2_2[m.trainIdx].pt)

    print("matches 1 with 0.7: " + str(len(good)))
    img_out=cv2.drawMatchesKnn(img1, keypoints_cv2_1, img2, keypoints_cv2_2, good, None)
    cv2.imwrite("out\\1_0,7.png", img_out)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #create FundamentalMatrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    #select only -----
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 3)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1, img2 = drawlines(gray1, gray2, lines, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(gray2, gray1, lines2, pts1, pts2)
    cv2.imshow('img', img1)
    cv2.waitKey(2000)
    cv2.imshow('img', img3)
    cv2.waitKey(2000)


matches_1()