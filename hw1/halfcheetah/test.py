import numpy as np
B = np.zeros((100,2,1))
A = np.random.randint(5, size=10)
print A
print B[A, :, :]