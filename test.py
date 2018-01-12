import numpy as np

st = [[1, 2, 3, 4, 5], [2,3,4,5,6]]
# st = np.asarray(st)
p = st[0][:]
print("old p ", p)

st[0] = [2,55,4,6,6]
print(p)
print(st)