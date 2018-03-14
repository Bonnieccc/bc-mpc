import numpy as np
import tensorflow as tf
import copy

from cost_functions import cheetah_cost_fn, trajectory_cost_fn

FLAGS = tf.app.flags.FLAGS

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
        if FLAGS.env_name == "HalfCheetah-v1":
            st = env.reset_model()
        else:
            st = env.reset()

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

def path_cost(cost_fn, path):
    # Utility to compute cost a path for a given cost function
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

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def traj_segment_generator(pi, mpc_controller, mpc_ppo_controller, bc_data_buffer, env, mpc, ppo_mpc, horizon):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    ob = env.reset()
    new = True # marks if we're on first timestep of an episode

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    nxt_obs = np.array([ob for _ in range(horizon)])

    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    mpcacs = acs.copy()

    print("using mpc: ", mpc)

    if mpc:
        if ppo_mpc:
            print("Using ppo mpc")
        else:
            print("Using normal mpc")

    while True:
        prevac = ac

        ac, vpred = pi.act(ob, stochastic=True)

        if mpc:
            if ppo_mpc:
                mpc_ac = mpc_ppo_controller.get_action(ob)
            else:
                mpc_ac = mpc_controller.get_action(ob)
        else:
            mpc_ac = copy.deepcopy(ac)

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac
        mpcacs[t] = mpc_ac

        ob, rew, done, _ = env.step(mpc_ac)
        new = False

        nxt_obs[t] = ob
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1

        t += 1


        # if t > 0 and t % (horizon-1) == 0:
        if t >= horizon:

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            print("ep_rets ", ep_rets)
            print("ep_lens ", ep_lens)

            break


    sec = {"ob" : obs, "rew" : rews, "nxt_ob": nxt_obs, "vpred" : vpreds, "new" : news,
    "ac" : acs, "prevac" : prevacs, "mpcac" : mpcacs, "nextvpred": vpred * (1 - new),
    "ep_rets" : ep_rets, "ep_lens" : ep_lens}

    
    return  sec

def add_vtarg_and_adv(seg, gamma=0.99, lam=0.95):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def policy_net_eval(sess, env, policy_net, env_horizon):
    print('---------- Policy Net Performance ---------')
    # st = env.reset_model()
    st = env.reset()

    returns = 0

    for j in range(env_horizon):
        at, vpred = policy_net.act(st, stochastic=False)
        # print(at)
        nxt_st, r, _, _ = env.step(at)
        st = nxt_st
        returns += r

    print("return: ", returns)

    return returns

