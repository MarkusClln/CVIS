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

    print("matches 1 with 0.8: " + str(len(good)))
    img_out=cv2.drawMatchesKnn(img1, keypoints_cv2_1, img2, keypoints_cv2_2, good, None)
    cv2.imwrite("out\\1_0,8.png", img_out)

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
    img3, img4 = drawlines(gray2, gray1, lines2, pts2, pts1)
    #cv2.imshow('img', img1)
    cv2.imwrite("out\\1_08.png", img1)
    #cv2.waitKey(8000)
    #cv2.imshow('img', img3)
    cv2.imwrite("out\\2_08.png", img3)
    #cv2.waitKey(8000)


def matches_2():
    descriptors_cv2_1 = to_cv2_di(descriptors3)
    descriptors_cv2_2 = to_cv2_di(descriptors4)

    keypoints_cv2_1 = to_cv2_kplist(detected_keypoints3)
    keypoints_cv2_2 = to_cv2_kplist(detected_keypoints4)


    bf = cv2.BFMatcher()

    img1 = cv2.imread(image_pathes[2])
    img2 = cv2.imread(image_pathes[3])

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

    print("matches 2 with 0.8: " + str(len(good)))
    img_out=cv2.drawMatchesKnn(img1, keypoints_cv2_1, img2, keypoints_cv2_2, good, None)
    cv2.imwrite("out\\.2_0,8png", img_out)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #create FundamentalMatrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    #fx = fy = 721.5
    #cx = 690.5
    #cy = 172.8
    #F[0][0] = fx
    #F[0][2] = cx
    #F[1][1] = fy
    #F[1][2] = cy
    # F = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 0]])

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
    img3, img4 = drawlines(gray2, gray1, lines2, pts2, pts1)
    #cv2.imshow('img', img1)
    cv2.imwrite("out\\3_08.png", img1)
    #cv2.waitKey(8000)
    #cv2.imshow('img', img3)
    cv2.imwrite("out\\4_08.png", img3)
    #cv2.waitKey(8000)


def Aufgabe_02():
    descriptors_cv2_1 = to_cv2_di(descriptors1)
    descriptors_cv2_2 = to_cv2_di(descriptors2)

    keypoints_cv2_1 = to_cv2_kplist(detected_keypoints1)
    keypoints_cv2_2 = to_cv2_kplist(detected_keypoints2)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors_cv2_1, descriptors_cv2_2, k=2)
    good = []
    pts1_real = []
    pts2_real = []
    theshold_matching = 0.7
    for m, n in matches:
        if m.distance < theshold_matching * n.distance:
            good.append([m])
            pts1_real.append(keypoints_cv2_1[m.queryIdx].pt)
            pts2_real.append(keypoints_cv2_2[m.trainIdx].pt)

    pts1 = np.float32(pts1_real)
    pts2 = np.float32(pts2_real)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    fx = fy = 721.5
    cx = 690.5
    cy = 172.8
    K = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    I= np.eye(3)
    O = np.zeros((3,1))



    P0 = K.dot(np.hstack((I,O)))
    E = K.T *np.mat(F)*K
    R1, R2, t = cv2.decomposeEssentialMat(E)

    P1_1 = K.dot(np.hstack((R1, t)))

    P2_1 = K * np.hstack((R1, -t))

    P3_1 = K * np.hstack((R2, t))

    P4_1 = K * np.hstack((R2, -t))


    #print(pts1)
    #print(pts2)
    #first_inliers = np.array(pts1).reshape(-1, 3)[:, :2]
    #second_inliers = np.array(pts2).reshape(-1, 3)[:, :2]

    #print(pts1)
    #pts1 = np.resize(pts1,(1,3))
    print(pts1)

    print(pts1.T)

    #pointcloud = cv2.triangulatePoints(P0, P1_1, pts1.T, pts2.T)
    #pointcloud = cv2.triangulatePoints(P2_1, P0, pts1.T, pts2.T)
    #pointcloud = cv2.triangulatePoints(P3_1, P0, pts1.T, pts2.T)
    #pointcloud = cv2.triangulatePoints(P4_1, P0, pts1.T, pts2.T)
    #print(pointcloud)


    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        end_header
        '''


    def write_ply(fn, verts):
        verts = verts.reshape(-1, 3)
        with open(fn, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f')

    #write_ply('out\\punktwolke.ply', pointcloud)

matches_1()
Aufgabe_02()
