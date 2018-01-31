import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

from policy_net import policy_network, policy_network_ppo
from value_net import value_network
from controllers import RandomController

from cheetah_env import HalfCheetahEnvNew
from utils import denormalize, normalize, pathlength


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
             trpo=False,
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

    policy_nn = policy_network(sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate)

    if nn_baseline:
        value_nn = value_network(sess, ob_dim, n_layers, size, learning_rate)

    sess.__enter__() # equivalent to `with sess:`

    tf.global_variables_initializer().run() #pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = policy_nn.predict(ob)
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])


      
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

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:

            b_n = value_nn.predict(ob_no)
            b_n = normalize(b_n)
            b_n = denormalize(b_n, np.std(q_n), np.mean(q_n))

            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:

            adv_n = normalize(adv_n)


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:

            b_n_target = normalize(q_n)

            value_nn.fit(ob_no, b_n_target)


        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        policy_nn.fit(ob_no, ac_na, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=1000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--trpo', action='store_true')

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
                trpo=args.trpo
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
