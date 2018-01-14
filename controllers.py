import numpy as np
from cost_functions import trajectory_cost_fn
import time
# from memory_profiler import profile
import copy
class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    # @profile
    def sample_random_actions(self):
      
        # sample random action trajectories
        actions = []
        for n in range(self.num_simulated_paths):
            for h in range(self.horizon):
                actions.append(self.env.action_space.sample())

        np_action_paths = np.asarray(actions)
        np_action_paths = np.reshape(np_action_paths, [self.horizon, self.num_simulated_paths, -1])

        return np_action_paths

    # @profile
    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        action_paths = self.sample_random_actions()

        # get init observations and copy num_simulated_paths times
        states = np.tile(state, [self.num_simulated_paths, 1])

        states_paths_all = []
        states_paths_all.append(states)

        for i in range(self.horizon):
            states = self.dyn_model.predict(states, action_paths[i, :, :])
            states_paths_all.append(states)

        # evaluate trajectories
        states_paths_all = np.asarray(states_paths_all)

        # batch cost function
        states_paths = states_paths_all[:-1, :, :]
        states_nxt_paths = states_paths_all[1:, :, :]

        costs = trajectory_cost_fn(self.cost_fn, states_paths, action_paths, states_nxt_paths)

        min_cost_path = np.argmin(costs)
        opt_cost = costs[min_cost_path]
        opt_action_path = action_paths[:, min_cost_path, :]
        opt_action = copy.copy(opt_action_path[0])

        # print("MPC imagine min cost: ", opt_cost)
        return opt_action


class MPCcontroller_BC(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model,
                 bc_network, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.bc_network = bc_network
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    # @profile
    def sample_random_actions(self):
      
        # sample random action trajectories
        actions = []
        for n in range(self.num_simulated_paths):
            for h in range(self.horizon):
                actions.append(self.env.action_space.sample())

        np_action_paths = np.asarray(actions)
        np_action_paths = np.reshape(np_action_paths, [self.horizon, self.num_simulated_paths, -1])

        return np_action_paths

    # @profile
    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        # action_paths = self.sample_random_actions()

        # get init observations and copy num_simulated_paths times
        states = np.tile(state, [self.num_simulated_paths, 1])

        states_paths_all = []
        states_paths_all.append(states)

        for i in range(self.horizon):
            actions = self.bc_network.predict(states)
            states = self.dyn_model.predict(states, actions)

            # states = self.dyn_model.predict(states, action_paths[i, :, :])
            states_paths_all.append(states)

        # evaluate trajectories
        states_paths_all = np.asarray(states_paths_all)

        # batch cost function
        states_paths = states_paths_all[:-1, :, :]
        states_nxt_paths = states_paths_all[1:, :, :]

        costs = trajectory_cost_fn(self.cost_fn, states_paths, action_paths, states_nxt_paths)

        min_cost_path = np.argmin(costs)
        opt_cost = costs[min_cost_path]
        opt_action_path = action_paths[:, min_cost_path, :]
        opt_action = copy.copy(opt_action_path[0])

        # print("MPC imagine min cost: ", opt_cost)
        return opt_action