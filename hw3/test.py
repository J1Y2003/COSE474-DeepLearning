import numpy as np

"""W = np.array([[4,3], [2,3], [4,2], [1,6]])
x = np.ones((20,2))
b = np.array([10,10,10,10])

test = np.matmul(W, np.transpose(x))
print(np.transpose(test) + b)
print(test.shape)"""

test1 = np.array([1,2,3,4])
print(test1.reshape((4,1)))