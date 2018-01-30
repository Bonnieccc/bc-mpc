import numpy as np
import tensorflow as tf
from cost_functions import cheetah_cost_fn, trajectory_cost_fn

# Utilities

def denormalize(n_data, std, mean):
    data = n_data * std + mean
    return data

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean)/(std + 1e-10)
    return data

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):

    with tf.variable_scope(scope):
        net = input_placeholder

        for n in range(n_layers-1):
            net = tf.layers.dense(net, 64, activation=activation)

        out = tf.layers.dense(net, output_size, activation=output_activation)

    return out
    
def pathlength(path):
    return len(path["reward"])


# Model based parts

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    """ YOUR CODE HERE """

    paths = []
    for i in range(num_paths):
        # print("random data iter ", i)
        st = env.reset_model()
        path = {'observations': [], 'actions': [], 'next_observations':[]}

        for t in range(horizon):
           at = controller.get_action(st)
           st_next, _, _, _ = env.step(at)

           path['observations'].append(st)
           path['actions'].append(at)
           path['next_observations'].append(st_next)
           st = st_next

        paths.append(path)

    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    # Nomalization statistics
    sample_state, sample_action, sample_nxt_state, sample_state_delta = data.sample(data.size)
    mean_obs = np.mean(sample_state, axis=0)
    mean_action = np.mean(sample_action, axis=0)
    mean_nxt_state = np.mean(sample_nxt_state, axis=0)
    mean_deltas = np.mean(sample_state_delta, axis=0)

    std_obs = np.std(sample_state, axis=0)
    std_action = np.std(sample_action, axis=0)
    std_nxt_state = np.std(sample_nxt_state, axis=0)
    std_deltas = np.std(sample_state_delta, axis=0)

    return [mean_obs, std_obs, mean_action, std_action, mean_nxt_state, std_nxt_state, mean_deltas, std_deltas]

