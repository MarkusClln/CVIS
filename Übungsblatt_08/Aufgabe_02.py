from siftdetector import detect_keypoints
import glob
import cv2
import numpy as np
import pickle

class FettaShiat():
    def __init__(self, K):
        with open('keypoints.pkl') as f:
            self.img1_kp_f, self.img1_ds_f, self.img2_kp_f, self.img2_ds_f, self.img3_kp_f, self.img3_ds_f, self.img4_kp_f, self.img4_ds_f = pickle.load(f)

        self.K = K

    def set_img(self, first_img_kp, first_img_ds, second_img_kp, second_img_ds):
        self.first_img_kp = first_img_kp
        self.first_img_ds = first_img_ds
        self.second_img_kp = second_img_kp
        self.second_img_ds = second_img_ds


    def to_cv2_kplist(self, kp):
        return list(map(self.to_cv2_kp, kp))

    def to_cv2_kp(self, kp):
        return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)

    def to_cv2_di(self, di):
        return np.asarray(di, np.float32)

    def find_good_points(self, theshold):
        img1_kp = self.to_cv2_kplist(self.first_img_kp)
        img1_ds = self.to_cv2_di(self.first_img_ds)
        img2_kp = self.to_cv2_kplist(self.second_img_kp)
        img2_ds = self.to_cv2_di(self.second_img_ds)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(img1_ds, img2_ds, k=2)

        self.good = []
        self.pts1 = []
        self.pts2 = []
        theshold_matching = theshold

        for m, n in matches:
            if m.distance < theshold_matching * n.distance:
                self.good.append([m])
                self.pts1.append(img1_kp[m.queryIdx].pt)
                self.pts2.append(img2_kp[m.trainIdx].pt)

        self.pts1 = np.float32(self.pts1)
        self.pts2 = np.float32(self.pts2)

    def find_fundamental_mat(self):
        self.F, self.mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_LMEDS)

    def find_essential_mat(self):
        self.E = self.K.T * np.mat(self.F) * self.K
        self.R1, self.R2, self.t = cv2.decomposeEssentialMat(self.E)


    def validate_points(self, pointcloud):
        counter = 0
        for x in pointcloud[2]:
            if(x<0):
                counter += 1
        return counter

    def create_pointcloud(self):
        I = np.eye(3)
        O = np.zeros((3, 1))

        P0 = K.dot(np.hstack((I, O)))
        P1 = K.dot(np.hstack((self.R1, self.t)))
        P2 = K.dot(np.hstack((self.R1, -self.t)))
        P3 = K.dot(np.hstack((self.R2, self.t)))
        P4 = K.dot(np.hstack((self.R2, -self.t)))

        X1 = cv2.triangulatePoints(P0, P1, self.pts1.T, self.pts2.T)
        X2 = cv2.triangulatePoints(P0, P2, self.pts1.T, self.pts2.T)
        X3 = cv2.triangulatePoints(P0, P3, self.pts1.T, self.pts2.T)
        X4 = cv2.triangulatePoints(P0, P4, self.pts1.T, self.pts2.T)

        x1_valid = self.validate_points(X1)
        x2_valid = self.validate_points(X2)
        x3_valid = self.validate_points(X3)
        x4_valid = self.validate_points(X4)

        list_valids =[x1_valid, x2_valid, x3_valid, x4_valid]
        min_value = min(list_valids)
        print(min_value)
        if(min_value == x1_valid):
            return X1
        elif(min_value == x2_valid):
            return X2
        elif(min_value==x3_valid):
            return X3
        elif(min_value==x4_valid):
            return X4

    def write_ply(self, fn, verts):
        ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                end_header
                '''
        print(verts)
        verts = verts.reshape(-1, 3)
        with open(fn, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f')
fx = 721.5
fy = 721.5
cx = 690.5
cy = 172.8

K = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float)

A = FettaShiat(K)
#A.set_img(A.img1_kp_f, A.img1_ds_f, A.img2_kp_f, A.img2_ds_f)
A.set_img(A.img3_kp_f, A.img3_ds_f, A.img4_kp_f, A.img4_ds_f)
A.find_good_points(0.7)
A.find_fundamental_mat()
A.find_essential_mat()
pointcloud = A.create_pointcloud()
A.write_ply('out\\punktwolke_pic2.ply', pointcloud)