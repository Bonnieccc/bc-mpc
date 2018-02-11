import numpy as np

class StateNode(object):
    def __init__(self, state, parent, depth):
        self.type = "state_node"
        self.state = state
        self.parent = parent
        self.children = []
        self.visited_times = 0
        self.cumulative_reward = 0.
        self.depth = depth

    def add_child(child_node):
        self.children.append(child_node)

    def value():
        value = self.cumulative_reward / self.visited_times
        return value

    def num_children():
        num_children = len(self.children)
        return num_children

class StateActionNode(object):
    def __init__(self, state, action, parent, depth):
        self.type = "state_action_node"
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visited_time = 0
        self.value = 0.
        self.depth = depth

    def add_child(child_node):
        self.children.append(child_node)

    def value():
        value = self.cumulative_reward / self.visited_times
        return value

    def num_children():
        num_children = len(self.children)
        return num_children