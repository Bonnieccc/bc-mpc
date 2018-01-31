import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
import copy
from multiprocessing import Process

from policy_net import policy_network, policy_network_ppo
# from value_net import value_network
from controllers import RandomController

from cheetah_env import HalfCheetahEnvNew
from utils import denormalize, normalize, pathlength, reward_to_q
from data_buffer import  DataBuffer_general


from baselines.common import Dataset

def traj_segment_generator(pi, env, horizon):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    ob = env.reset()
    new = False # marks if we're on first timestep of an episode

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac = pi.predict(ob)
        vpred = pi.baseline_predict(ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            print("ep_lens ", ep_lens)
            sec = copy.deepcopy({"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
            "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
            "ep_rets" : ep_rets, "ep_lens" : ep_lens})
            yield sec
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, done, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        # print('cur_ep_len', cur_ep_len)
        # print('done', done)
        # print('new', new)

        if done or (t%1000 == 0 and t>0):
            ob = env.reset()
            new = True 

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            new = False # marks if we're on first timestep of an episode


        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
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

#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name=' HalfCheetah',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=False, 
             animate=True, 
             logdir=None, 
             normalize_advantages=False,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = HalfCheetahEnvNew()
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # Print environment infomation
    print("Environment name: ",  "HalfCheetah")
    print("Action space is discrete: ", discrete)
    print("Action space dim: ", ac_dim)
    print("Observation space dim: ", ob_dim)
    print("Max_path_length ", max_path_length)


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#


    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=4) 

    sess = tf.Session(config=tf_config)

    policy_nn = policy_network_ppo(sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate)

    # if nn_baseline:
    #     value_nn = value_network(sess, ob_dim, n_layers, size, learning_rate)

    sess.__enter__() # equivalent to `with sess:`

    tf.global_variables_initializer().run() #pylint: disable=E1101




    data_buffer_ppo = DataBuffer_general(10000, 4)
    seg_gen = traj_segment_generator(policy_nn, env, min_timesteps_per_batch)


    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)


        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, 0.95)

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#


        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=False)

        for n in range(len(ob)):
            data_buffer_ppo.add((ob[n], ac[n], atarg[n], tdlamret[n]))

        print("data_buffer_ppo", data_buffer_ppo.size)


        optim_batchsize = 64
        optim_epochs = 10

        # print("sample_ob_no", sample_ob_no.shape)
        # print("sample_ac_na", sample_ac_na.shape)
        # print("sample_adv_n", sample_adv_n.shape)

        # print("sample_ob_no", sample_ob_no)
        # print("sample_ac_na", sample_ac_na)
        # print("sample_adv_n", sample_adv_n)

        # policy_nn.fit(ob_no, ac_na, adv_n)
        policy_nn.assign_old_eq_new()


        for _ in range(optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                policy_nn.fit(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])

        # for i in range(optim_epochs*32):
        #     sample_ob_no, sample_ac_na, sample_adv_n, sample_b_n_target = data_buffer_ppo.sample(optim_batchsize)

        #     policy_nn.fit(sample_ob_no, sample_ac_na, sample_adv_n, sample_b_n_target)



        data_buffer_ppo.clear()

        # Log diagnostics
        # returns = [path["reward"].sum() for path in paths]
        # ep_lengths = [pathlength(path) for path in paths]

        ep_lengths = seg["ep_lens"]
        returns =  seg["ep_rets"]

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=2048)
    parser.add_argument('--ep_len', '-ep', type=float, default=1000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
