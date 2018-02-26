import numpy as np
import random
from collections import deque

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


class DataBufferGeneral(object):
    def __init__(self, buffer_size, item_num):
        self.buffer = deque()
        self.size = 0
        self.buffer_size = buffer_size
        self.item_num = item_num

    def add(self, data_items):
        if self.size <= self.buffer_size:
            self.buffer.append(data_items)
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(data_items)


    def sample(self, num):
        # Sample N
        batch = []
        if self.size < num:
            batch = random.sample(self.buffer, self.size)
        else:
            batch = random.sample(self.buffer, num)

        return_data = []
        for i in range(self.item_num):
            return_data.append(np.array([_[i] for _ in batch]))

        return return_data

    def clear(self):
        self.buffer.clear()
        self.size = 0

