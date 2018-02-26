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

def reward_to_q(paths, gamma, reward_to_go=False):
    # # Other's Code
    # q_n = []
    # for path in paths:
    #     q = 0
    #     q_path = []

    #     # Dynamic programming over reversed path
    #     for rew in reversed(path["reward"]):
    #         q = rew + gamma * q
    #         q_path.append(q)
    #     q_path.reverse()

    #     # Append these q values
    #     if not reward_to_go:
    #         q_path = [q_path[0]] * len(q_path)
    #     q_n.extend(q_path)


    # YOUR_CODE_HERE
    if reward_to_go:
        q_n = []
        for path in paths:
            for t in range(len(path["reward"])):
                t_ = 0
                q = 0
                while t_ < len(path["reward"]):
                    if t_ >= t:
                        q += gamma**(t_-t) * path["reward"][t_]
                    t_ += 1
                q_n.append(q)
        q_n = np.asarray(q_n)

    else:
        q_n = []
        for path in paths:
            for t in range(len(path["reward"])):
                t_ = 0
                q = 0
                while t_ < len(path["reward"]):
                    q += gamma**t_ * path["reward"][t_]
                    t_ += 1
                q_n.append(q)
        q_n = np.asarray(q_n)
    return q_n
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
        path = {'observations': [], 'actions': [], 'rewards': [], 'next_observations':[]}

        for t in range(horizon):
           at = controller.get_action(st)
           st_next, r_t, _, _ = env.step(at)

           path['observations'].append(st)
           path['actions'].append(at)
           path['rewards'].append(r_t)
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
    Return 10 elements: mean, std
    """

    """ YOUR CODE HERE """
    # Nomalization statistics
    sample_state, sample_action, sample_reward, sample_nxt_state, sample_state_delta = data.sample(data.size)

    mean_obs = np.mean(sample_state, axis=0)
    mean_action = np.mean(sample_action, axis=0)
    mean_reward = np.mean(sample_reward, axis=0)
    mean_nxt_state = np.mean(sample_nxt_state, axis=0)
    mean_deltas = np.mean(sample_state_delta, axis=0)

    std_obs = np.std(sample_state, axis=0)
    std_action = np.std(sample_action, axis=0)
    std_reward = np.std(sample_reward, axis=0)
    std_nxt_state = np.std(sample_nxt_state, axis=0)
    std_deltas = np.std(sample_state_delta, axis=0)

    return [mean_obs, std_obs, mean_action, std_action, mean_reward, std_reward, mean_nxt_state, std_nxt_state, mean_deltas, std_deltas]

