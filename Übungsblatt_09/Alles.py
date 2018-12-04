from siftdetector import detect_keypoints
import glob
import cv2
import numpy as np
import pickle

class creator():
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
        self.img1_kp = self.to_cv2_kplist(self.first_img_kp)
        self.img1_ds = self.to_cv2_di(self.first_img_ds)
        self.img2_kp = self.to_cv2_kplist(self.second_img_kp)
        self.img2_ds = self.to_cv2_di(self.second_img_ds)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.img1_ds, self.img2_ds, k=2)

        self.good = []
        self.pts1 = []
        self.pts2 = []
        theshold_matching = theshold

        for m, n in matches:
            if m.distance < theshold_matching * n.distance:
                self.good.append([m])
                self.pts1.append(self.img1_kp[m.queryIdx].pt)
                self.pts2.append(self.img2_kp[m.trainIdx].pt)

        self.pts1 = np.float32(self.pts1)
        self.pts2 = np.float32(self.pts2)

    def find_fundamental_mat(self):
        self.F, self.mask = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_LMEDS)

    def find_essential_mat(self):
        self.E = self.K.T * np.mat(self.F) * self.K
        self.R1, self.R2, self.t = cv2.decomposeEssentialMat(self.E)


    def validate_points(self, pointcloud):
        counter = 0
        for x in pointcloud[2][0]:
            if(x<0):
                counter += 1
        return counter

    def drawlines(self, img1, img2, lines, pts1, pts2):
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def draw_Matches(self, path_1, path_2):
        img1 = cv2.imread(path_1)
        img2 = cv2.imread(path_2)
        img_out = cv2.drawMatchesKnn(img1, self.img1_kp, img2, self.img2_kp, self.good, None)
        cv2.imshow('img', img_out)
        cv2.waitKey(8000)

    def draw_epi(self, path_1, path_2):
        img1 = cv2.imread(path_1)
        img2 = cv2.imread(path_2)
        pts1 = self.pts1[self.mask.ravel() == 1]
        pts2 = self.pts2[self.mask.ravel() == 1]
        lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, self.F)
        lines = lines.reshape(-1, 3)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1, img2 = self.drawlines(gray1, gray2, lines, pts1, pts2)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, self.F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.drawlines(gray2, gray1, lines2, pts2, pts1)
        cv2.imshow('img', img1)
        cv2.waitKey(8000)
        cv2.imshow('img', img3)
        cv2.waitKey(8000)

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

        X1 = cv2.convertPointsFromHomogeneous(X1.T)
        X2 = cv2.convertPointsFromHomogeneous(X2.T)
        X3 = cv2.convertPointsFromHomogeneous(X3.T)
        X4 = cv2.convertPointsFromHomogeneous(X4.T)

        x1_valid = self.validate_points(X1.T)
        x2_valid = self.validate_points(X2.T)
        x3_valid = self.validate_points(X3.T)
        x4_valid = self.validate_points(X4.T)


        list_valids =[x1_valid, x2_valid, x3_valid, x4_valid]
        min_value = min(list_valids)
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
        verts = verts.reshape(-1, 3)
        with open(fn, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f')

    def write_plyC(self, fn, verts):
        ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
        verts = verts.reshape(-1, 6)
        with open(fn, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')


    def depthmap(self, path1, path2):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        blockSize = 10  # Gibt Fenstergroesse an (1-20)
        min_disp = 1  # Gibt minimale Disparitaet an (0-10)
        y = 4
        num_disp = 16 * y  # Gibt maxinale Disparitaet an

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=blockSize, speckleWindowSize = 100, speckleRange = 1 )

        disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
        disparity_max = disparity.max()
        disparity_q = disparity / disparity_max
        disparity_q = disparity_q * 255
        img = disparity_q.astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        #cv2.imshow("Window", img_color)

        pic_out = np.zeros((len(disparity), len(disparity[0]), 6), np.float)

        for n in range(0, len(disparity)):
            for m in range(0, len(disparity[n])):
                pic_out[n][m][0]=n
                pic_out[n][m][1] =m
                pic_out[n][m][2] = self.get_Z(disparity_q[n][m])
                #pic_out[n][m][2] = disparity_q[n][m]
                pic_out[n][m][3] = img1[n][m][2]
                pic_out[n][m][4] = img1[n][m][1]
                pic_out[n][m][5] = img1[n][m][0]

        return pic_out


    def get_Z(self, value):
        if(value == 0):
            return 0
        else:
            #out = (self.K.item(0) * self.t.item(0)) / value
            out = (721.5 * self.t.item(0)) / value
            #out = 100 - (0.54 * self.K.item(0)) / value
            #print(out)
            return out * -1

    def extrinsische_transformation(self, pointcloud):
        t1 = np.array([0.3, 0.3, 0.3])
        t2 = np.array([0.6, 0.6, 0.6])
        t3 = np.array([0.9, 0.9, 0.9])
        t4 = np.array([1.2, 1.2, 1.2])

        self.extrinsics = np.vstack((np.hstack((self.R1, np.reshape(t4, (3,1)))), np.array([0., 0., 0., 1.])))

        pic_out = np.zeros((len(pointcloud), len(pointcloud[0]), 6), np.float)
        for n in range(len(pointcloud)):
            for m in range(len(pointcloud[0])):
                x = np.array([[pointcloud[n][m][0]], [pointcloud[n][m][1]], [pointcloud[n][m][2]]])
                x_trans = self.to3D(x)
                pic_out[n][m][0] = x_trans[0]
                pic_out[n][m][1] = x_trans[1]
                pic_out[n][m][2] = x_trans[2]
                pic_out[n][m][3] = pointcloud[n][m][3]
                pic_out[n][m][4] = pointcloud[n][m][4]
                pic_out[n][m][5] = pointcloud[n][m][5]

        return pic_out



    def to3D(self, point):
        x = np.vstack([point, [1]])
        x3D = self.extrinsics.dot(x)
        x3D = np.array([x3D[0] / x3D[3], x3D[1] / x3D[3], x3D[2] / x3D[3]])
        return x3D

    def to2D(self, point):
        t1 = np.array([0, 0, 0])
        t1 = np.reshape(t1, (3, 1))
        x = np.vstack([point, [1]])
        P = np.hstack([self.K, t1])
        x2D = P.dot(x)
        if(x2D[2] != 0):
            x2D = np.array([x2D[0] / x2D[2], x2D[1] / x2D[2]])
            x2D = np.reshape(x2D, (2,1))
            return x2D
        else:
            return (np.array([x2D[0] / 1, x2D[1] / 1]))

    def pointcloud_to_pic(self, pointcloud):
        for n in range(len(pointcloud)):
            for m in range(len(pointcloud[0])):
                x = np.array([[pointcloud[n][m][0]], [pointcloud[n][m][1]], [pointcloud[n][m][2]]])
                x_trans = self.to2D(x)
                pointcloud[n][m][0] = x_trans[0]
                pointcloud[n][m][1] = x_trans[1]
        x_min = np.amin(pointcloud[:][:][0])
        y_min = np.amin(pointcloud[:][:][1])
        print (np.amax(pointcloud[:][:][0]))
        print( x_min)
        print(np.amax(pointcloud[:][:][1]))
        print(y_min)
        pic_out = np.zeros((np.amax(pointcloud[:][:][0]) + 1 - x_min, np.amax(pointcloud[:][:][1]) + 1 - y_min, 3), np.uint8)
        for n in range(len(pointcloud)):
            for m in range(len(pointcloud[0])):
                pic_out[int(x_trans[0]-x_min)][int(x_trans[1]-y_min)] = pointcloud[n][m][5]
                pic_out[int(x_trans[0] - x_min)][int(x_trans[1] - y_min)] = pointcloud[n][m][4]
                pic_out[int(x_trans[0] - x_min)][int(x_trans[1] - y_min)] = pointcloud[n][m][3]
        return pic_out

    def pointcloud_to_pic_2(self, pointcloud):
        #points_3D = np.array([pointcloud[0][0][0], pointcloud[0][0][1], pointcloud[0][0][2]])

        points_x = []
        points_y = []
        points_z = []
        for n in range(len(pointcloud)):
            for m in range(len(pointcloud[0])):
                points_x.append(pointcloud[n][m][0])
                points_y.append(pointcloud[n][m][1])
                points_z.append(pointcloud[n][m][2])
        color_r = []
        color_g = []
        color_b = []
        for n in range(len(pointcloud)):
            for m in range(len(pointcloud[0])):
                color_r.append(pointcloud[n][m][3])
                color_g.append(pointcloud[n][m][4])
                color_b.append(pointcloud[n][m][5])
        color_r = np.array(color_r, dtype=np.uint8)
        color_g = np.array(color_g, dtype=np.uint8)
        color_b = np.array(color_b, dtype=np.uint8)

        points_3D = np.concatenate(([points_x], [points_y], [points_z]), axis=0).T
        rvec = np.array([0.,0.,0.])
        points_2D, jac= cv2.projectPoints(points_3D,rvec,rvec, self.K, distCoeffs= None)

        points_2D = np.reshape(points_2D,(-1,2))

        points_2D = points_2D.T
        print(max(points_2D[0]))
        print(max(points_2D[1]))
        points_2D[0]= points_2D[0]/max(points_2D[0]) * 374
        points_2D[1] = points_2D[1] / max(points_2D[1]) * 1241
        points_2D = points_2D.T


        pic_out = np.zeros((374, 1241, 3),np.uint8)

        for i in range(0, len(points_2D)):
            if(int(points_2D[i][0])<374 and int(points_2D[i][1])<1241 ):
                pic_out[int(points_2D[i][0])][int(points_2D[i][1])] = [color_b[i], color_g[i], color_r[i]]

        cv2.imshow("Window", pic_out)
        cv2.imwrite("out\\test.png", pic_out)
        cv2.waitKey(8000)





image_pathes = glob.glob("images\*.png")
fx = 721.5
fy = 721.5
cx = 690.5
cy = 172.8
baseline = 0.54

K = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float)

A = creator(K)
#A.set_img(A.img1_kp_f, A.img1_ds_f, A.img2_kp_f, A.img2_ds_f)
A.set_img(A.img3_kp_f, A.img3_ds_f, A.img4_kp_f, A.img4_ds_f)
A.find_good_points(0.7)
A.find_fundamental_mat()
A.find_essential_mat()
#pointcloud = A.create_pointcloud()
#A.write_ply('out\\punktwolke_pic2.ply', pointcloud)

#A.draw_Matches(image_pathes[0], image_pathes[1])
#A.draw_epi(image_pathes[0], image_pathes[1])

pointcloud = A.depthmap(image_pathes[0], image_pathes[1])

p1 = A.extrinsische_transformation(pointcloud)


A.write_plyC('out\\punktwolke_4.ply', p1)


A.pointcloud_to_pic_2(p1)
