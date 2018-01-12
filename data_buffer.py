import numpy as np

class DataBuffer(object):
    def __init__(self):
        self.state_t = []
        self.action_t = []
        self.state_next_t = []
        self.size = 0

    def add(self, state_t, action_t, state_next_t):
        self.state_t.append(state_t)
        self.action_t.append(action_t)
        self.state_next_t.append(state_next_t)
        self.size = len(self.state_t)

    def sample(self, num):
        # Sample N
        assert (num <= self.size)
        sample_index = np.random.choice(self.size, num)
        sample_state = np.asarray([self.state_t[i] for i in sample_index])
        sample_action = np.asarray([self.action_t[i] for i in sample_index])
        sample_state_next_t = np.asarray([self.state_next_t[i] for i in sample_index])
        sample_state_delta = sample_state_next_t - sample_state
        return sample_state, sample_action, sample_state_next_t, sample_state_delta
