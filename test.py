import numpy as np


ac = np.random.rand(100,6)

print(ac.shape)
print("std ", np.std(ac))
print("std ", np.std(ac, axis=0))