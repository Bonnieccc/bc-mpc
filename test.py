import numpy as np

a = np.zeros([3, 5])
b = a[None]
print(a.shape)
print(b.shape)