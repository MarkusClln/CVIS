from siftdetector import detect_keypoints
import glob
import cv2
import numpy as np
import pickle

class pointcloud():

    def to_cv2_kp(self, kp):
        return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)

    def to_cv2_kplist(self, kp):
        return list(map(self.to_cv2_kp, kp))

    def to_cv2_di(self, di):
        return np.asarray(di, np.float32)


    def __init__(self, K):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.image_pathes = glob.glob("images\*.png")
        with open('keypoints.pkl') as f:
            self.detected_keypoints1, self.descriptors1, self.detected_keypoints2, self.descriptors2, self.detected_keypoints3, self.descriptors3, self.detected_keypoints4, self.descriptors4 = pickle.load(
                f)
        self.descriptors_cv2_1 = self.to_cv2_di(self.descriptors1)
        self.descriptors_cv2_2 = self.to_cv2_di(self.descriptors2)

        self.keypoints_cv2_1 = self.to_cv2_kplist(self.detected_keypoints1)
        self.keypoints_cv2_2 = self.to_cv2_kplist(self.detected_keypoints2)

        self.descriptors_cv2_3 = self.to_cv2_di(self.descriptors1)
        self.descriptors_cv2_4 = self.to_cv2_di(self.descriptors2)

        self.keypoints_cv2_3 = self.to_cv2_kplist(self.detected_keypoints1)
        self.keypoints_cv2_4 = self.to_cv2_kplist(self.detected_keypoints2)



    def _match_points(self, num1, num2):
        bf = cv2.BFMatcher()
        img1 = cv2.imread(self.image_pathes[num1])
        img2 = cv2.imread(self.image_pathes[num2])
        if(num1==1 and num2 ==2):
            matches = bf.knnMatch(self.descriptors_cv2_1, self.descriptors_cv2_2, k=2)
        elif (num1==3 and num2 == 4):
            matches = bf.knnMatch(self.descriptors_cv2_3, self.descriptors_cv2_4, k=2)
        self.good = []
        self.match_pts1 = []
        self.match_pts2 = []
        theshold_matching = 0.7
        for m, n in matches:
            if m.distance < theshold_matching * n.distance:
                self.good.append([m])
                if (num1 == 1 and num2 == 2):
                    matches = bf.knnMatch(self.descriptors_cv2_1, self.descriptors_cv2_2, k=2)
                    self.match_pts1.append(self.keypoints_cv2_1[m.queryIdx].pt)
                    self.match_pts2.append(self.keypoints_cv2_2[m.trainIdx].pt)
                elif (num1 == 3 and num2 == 4):
                    matches = bf.knnMatch(self.descriptors_cv2_3, self.descriptors_cv2_4, k=2)
                    self.match_pts1.append(self.keypoints_cv2_3[m.queryIdx].pt)
                    self.match_pts2.append(self.keypoints_cv2_4[m.trainIdx].pt)
        self.match_pts1 = np.int32(self.match_pts1)
        self.match_pts2 = np.int32(self.match_pts2)

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],
                                                     self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0],
                                                      self.match_pts2[i][1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]

        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]

            if not self._in_front_of_both_cameras(first_inliers,
                                                  second_inliers, R, T):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        """Determines whether point correspondences are in front of both
           images"""
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True
    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],
                                                     self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0],
                                                      self.match_pts2[i][1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]

        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]

            if not self._in_front_of_both_cameras(first_inliers,
                                                  second_inliers, R, T):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _plot_point_cloud(self, num1, num2):
        self._match_points(num1, num2)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

fx = fy = 721.5
cx = 690.5
cy = 172.8
K = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
module = pointcloud(K)
module._plot_point_cloud(1,2)

