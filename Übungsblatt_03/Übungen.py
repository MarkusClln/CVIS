import cv2
import numpy as np

a = np.array([2, 3, 4.1])
b = np.array([[1, 2, 3], [4, 5, 6]])

zeros = np.zeros((2, 3))
ones = np.ones((3, 4))
eye = np.eye(5)

ones *= 255

arange = np.arange(30)
arange = arange.reshape((5, 6))

d = np.arange(6)
e = np.arange(6)

e *= 10

R = np.eye(3)
print(R)
t = np.zeros((3, 1))
print(t)
Rt = np.hstack((R, t))


print(Rt)


