import numpy as np

state = np.zeros([3, 5])
print("state.shape", state.shape)

states = np.tile(state, [1000, 1])
print("states.shape", states.shape)

