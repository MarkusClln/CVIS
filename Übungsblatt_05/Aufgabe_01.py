import cv2
import numpy as np

x1 = np.array([[10], [10], [100]])
x2 = np.array([[33], [22], [111]])
x3 = np.array([[100], [100], [1000]])
x4 = np.array([[20], [-100], [100]])


def to2D(x):
    P = np.array([[460, 0, 320, 0], [0, 460, 240, 0], [0, 0, 1, 0]])
    x = np.vstack([x,[1]])
    x2D = P.dot(x)
    x2D = np.array([x2D[0]/x2D[2],[x2D[1]/x2D[2]]])
    return x2D


x1_2D = to2D(x1)
x2_2D = to2D(x2)
x3_2D = to2D(x3)
x4_2D = to2D(x4)

print("x1:  x:"+str(x1_2D[0])+", y:"+str(x1_2D[1]))
print("x2:  x:"+str(x2_2D[0])+", y:"+str(x2_2D[1]))
print("x3:  x:"+str(x3_2D[0])+", y:"+str(x3_2D[1]))
print("x4:  x:"+str(x4_2D[0])+", y:"+str(x4_2D[1]))


rvec = np.array([0,0,0], np.float)
tvec = np.array([0,0,0], np.float)

P2 = np.array([[460,0,320],[0,460,240],[0,0,1]], np.float)
points = np.array([[10,10,100], [33,22,111], [100,100,1000],[20,-100,100]],np.float)

test = cv2.projectPoints(points, rvec, tvec, P2, None)


for n in range(len(points)):
    print points[n], '==>', np.rint(test[0][n])

