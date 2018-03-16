import numpy as np
from cost_functions import trajectory_cost_fn
import time
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

    def sample_random_actions(self):
      
        # sample random action trajectories
        actions = []
        for n in range(self.num_simulated_paths):
            for h in range(self.horizon):
                actions.append(self.env.action_space.sample())

        np_action_paths = np.asarray(actions)
        np_action_paths = np.reshape(np_action_paths, [self.horizon, self.num_simulated_paths, -1])

        return np_action_paths

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

class MPCcontrollerReward(Controller):
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

    def sample_random_actions(self):
      
        # sample random action trajectories
        actions = []
        for n in range(self.num_simulated_paths):
            for h in range(self.horizon):
                actions.append(self.env.action_space.sample())

        np_action_paths = np.asarray(actions)
        np_action_paths = np.reshape(np_action_paths, [self.horizon, self.num_simulated_paths, -1])

        return np_action_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        action_paths = self.sample_random_actions()

        # get init observations and copy num_simulated_paths times
        states = np.tile(state, [self.num_simulated_paths, 1])

        # states_paths_all = []
        rewards_all = []
        # states_paths_all.append(states)

        for i in range(self.horizon):
            states, reward = self.dyn_model.predict(states, action_paths[i, :, :])
            # states_paths_all.append(states)
            rewards_all.append(reward)

        # # evaluate trajectories
        # states_paths_all = np.asarray(states_paths_all)

        # # batch cost function
        # states_paths = states_paths_all[:-1, :, :]
        # states_nxt_paths = states_paths_all[1:, :, :]

        # costs = trajectory_cost_fn(self.cost_fn, states_paths, action_paths, states_nxt_paths)

        rewards_all = np.asarray(rewards_all)
        rewards_all = np.sum(rewards_all, axis=0)
        rewards_all = np.reshape(rewards_all, [-1])
        min_cost_path = np.argmax(rewards_all)
        opt_imgreward = rewards_all[min_cost_path]
        opt_action_path = action_paths[:, min_cost_path, :]
        opt_action = copy.copy(opt_action_path[0])

        # print("MPC imagine min cost: ", opt_imgreward)
        return opt_action

class MPCcontrollerPolicyNet(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model,
                 policy_net, 
                 explore=1.,
                 self_exp=True,
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.policy_net = policy_net
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.self_exp = self_exp
        self.explore = explore

    def sample_random_actions(self):
      
        # sample random action trajectories
        np_action_paths = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high , size=[self.horizon, self.num_simulated_paths, len(self.env.action_space.high)])

        return np_action_paths


    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        exploration = self.sample_random_actions()

        # get init observations and copy num_simulated_paths times
        states = np.tile(state, [self.num_simulated_paths, 1])

        states_paths_all = []
        action_paths = []
        states_paths_all.append(states)

        for i in range(self.horizon):
            if self.self_exp:
                actions, _ = self.policy_net.act(states, stochastic=True)
            else:
                actions, _ = self.policy_net.act(states, stochastic=False)
                # actions += np.random.rand(self.num_simulated_paths, self.env.action_space.shape[0]) * (2*self.explore) - self.explore

                actions = (1 - self.explore) * actions + self.explore * exploration[i, :, :]

            states = self.dyn_model.predict(states, actions)

            # states = self.dyn_model.predict(states, action_paths[i, :, :])
            states_paths_all.append(states)
            action_paths.append(actions)

        # evaluate trajectories
        states_paths_all = np.asarray(states_paths_all)
        action_paths = np.asarray(action_paths)


        # batch cost function
        states_paths = states_paths_all[:-1, :, :]
        states_nxt_paths = states_paths_all[1:, :, :]

        # print("action_paths: ", action_paths.shape)
        # print("states_paths: ", states_paths.shape)
        # print("states_nxt_paths: ", states_nxt_paths.shape)

        costs = trajectory_cost_fn(self.cost_fn, states_paths, action_paths, states_nxt_paths)

        min_cost_path = np.argmin(costs)
        opt_cost = costs[min_cost_path]
        opt_action_path = action_paths[:, min_cost_path, :]
        opt_action = copy.copy(opt_action_path[0])

        # print("MPC imagine min cost: ", opt_cost)
        return opt_action

class MPCcontrollerPolicyNetReward(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model,
                 policy_net, 
                 explore=1.,
                 self_exp=True,
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.policy_net = policy_net
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.self_exp = self_exp
        self.explore = explore


    def sample_random_actions(self):
      
        # sample random action trajectories
        np_action_paths = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high , size=[self.horizon, self.num_simulated_paths, len(self.env.action_space.high)])

        return np_action_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        exploration = self.sample_random_actions()

        # get init observations and copy num_simulated_paths times
        states = np.tile(state, [self.num_simulated_paths, 1])
        states_paths_all = []
        action_paths = []
        states_paths_all.append(states)

        rewards_all = []
        action_paths = []

        for i in range(self.horizon):
            if self.self_exp:
                actions, _ = self.policy_net.act(states, stochastic=True)
            else:
                actions, _ = self.policy_net.act(states, stochastic=False)
                # actions += np.random.rand(self.num_simulated_paths, self.env.action_space.shape[0]) * (2*self.explore) - self.explore
                actions = (1 - self.explore) * actions + self.explore * exploration[i, :, :]


            states, reward = self.dyn_model.predict(states, actions)

            # states = self.dyn_model.predict(states, action_paths[i, :, :])
            states_paths_all.append(states)
            action_paths.append(actions)
            rewards_all.append(reward)

        # evaluate trajectories
        action_paths = np.asarray(action_paths)
        rewards_all = np.asarray(rewards_all)

        rewards_all = np.sum(rewards_all, axis=0)
        rewards_all = np.reshape(rewards_all, [-1])
        max_reward_path = np.argmax(rewards_all)
        opt_imgreward = rewards_all[max_reward_path]
        opt_action_path = action_paths[:, max_reward_path, :]
        opt_action = copy.copy(opt_action_path[0])

        return opt_action