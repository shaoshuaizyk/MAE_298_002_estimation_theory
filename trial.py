import numpy as np


A = np.random.rand(2, 3)
B = np.random.rand(3, 4)
C = A @ B
D = np.matmul(A, B)
E = np.dot(A, B)
print(C)
print(D)
print(E)