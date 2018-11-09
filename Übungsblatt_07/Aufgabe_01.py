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



def draw_keypoints():
    keypoints_cv2_1 = to_cv2_kplist(detected_keypoints1)
    keypoints_cv2_2 = to_cv2_kplist(detected_keypoints2)
    keypoints_cv2_3 = to_cv2_kplist(detected_keypoints3)
    keypoints_cv2_4 = to_cv2_kplist(detected_keypoints4)

    pic = cv2.imread(image_pathes[0])
    gray1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    out = cv2.drawKeypoints(gray1, keypoints_cv2_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', out)
    cv2.waitKey(500)
    cv2.imwrite("out\\img1.png",out)
    print("img1: "+str(len(keypoints_cv2_1)))

    pic = cv2.imread(image_pathes[1])
    gray1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    out = cv2.drawKeypoints(gray1, keypoints_cv2_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', out)
    cv2.waitKey(500)
    cv2.imwrite("out\\img2.png", out)
    print("img2: " + str(len(keypoints_cv2_2)))

    pic = cv2.imread(image_pathes[2])
    gray1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    out = cv2.drawKeypoints(gray1, keypoints_cv2_3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', out)
    cv2.waitKey(500)
    cv2.imwrite("out\\img3.png", out)
    print("img3: " + str(len(keypoints_cv2_3)))

    pic = cv2.imread(image_pathes[3])
    gray1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    out = cv2.drawKeypoints(gray1, keypoints_cv2_4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', out)
    cv2.waitKey(500)
    cv2.imwrite("out\\img4.png", out)
    print("img4: " + str(len(keypoints_cv2_4)))


def matches_1():
    descriptors_cv2_1 = to_cv2_di(descriptors1)
    descriptors_cv2_2 = to_cv2_di(descriptors2)

    keypoints_cv2_1 = to_cv2_kplist(detected_keypoints1)
    keypoints_cv2_2 = to_cv2_kplist(detected_keypoints2)


    bf = cv2.BFMatcher()
    matches=bf.match(descriptors_cv2_1, descriptors_cv2_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img1 = cv2.imread(image_pathes[0])
    img2 = cv2.imread(image_pathes[1])

    img_out = cv2.drawMatches(img1, keypoints_cv2_1, img2, keypoints_cv2_2, matches, None)
    cv2.imwrite("out\\1_all.png", img_out)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[0:30]
    img_out =cv2.drawMatches(img1, keypoints_cv2_1, img2, keypoints_cv2_2, matches, None)
    cv2.imwrite("out\\1_30.png", img_out)

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

    img_out=cv2.drawMatchesKnn(img1, keypoints_cv2_1, img2, keypoints_cv2_2, good, None)
    cv2.imwrite("out\\1_0,7.png", img_out)


def matches_2():
    descriptors_cv2_3 = to_cv2_di(descriptors3)
    descriptors_cv2_4 = to_cv2_di(descriptors4)

    keypoints_cv2_3 = to_cv2_kplist(detected_keypoints3)
    keypoints_cv2_4 = to_cv2_kplist(detected_keypoints4)

    bf = cv2.BFMatcher()
    matches=bf.match(descriptors_cv2_3, descriptors_cv2_4)
    matches = sorted(matches, key=lambda x: x.distance)

    img1 = cv2.imread(image_pathes[2])
    img2 = cv2.imread(image_pathes[3])

    img_out = cv2.drawMatches(img1, keypoints_cv2_3, img2, keypoints_cv2_4, matches, None)
    cv2.imwrite("out\\2_all.png", img_out)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[0:30]
    img_out =cv2.drawMatches(img1, keypoints_cv2_3, img2, keypoints_cv2_4, matches, None)
    cv2.imwrite("out\\2_30.png", img_out)

    matches = bf.knnMatch(descriptors_cv2_3,descriptors_cv2_4, k=2)
    good = []
    pts1 = []
    pts2 = []
    theshold_matching = 0.7
    for m,n in matches:
        if m.distance < theshold_matching*n.distance:
            good.append([m])
            pts1.append(keypoints_cv2_3[m.queryIdx].pt)
            pts2.append(keypoints_cv2_4[m.trainIdx].pt)

    img_out=cv2.drawMatchesKnn(img1, keypoints_cv2_3, img2, keypoints_cv2_4, good, None)
    cv2.imwrite("out\\2_0,7.png", img_out)

matches_2()