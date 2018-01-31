import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

from utils import denormalize, normalize, build_mlp, pathlength
from utils import sample, path_cost, compute_normalization

from policy_net import policy_network_mpc, policy_network_ppo
from value_net import value_network


from dynamics import NNDynamicsModel
from cheetah_env import HalfCheetahEnvNew
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
from controllers import MPCcontroller_learned_reward, MPCcontroller, RandomController, MPCcontroller_BC_learned_reward
from data_buffer import DataBuffer, DataBuffer_SA, DataBuffer_general


MPC = False
PG = True


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(
             exp_name='',
             env_name='',
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

             # mb mpc arguments
             model_learning_rate=1e-3,
             onpol_iters=10,
             dynamics_iters=260,
             batch_size=512,
             num_paths_random=10, 
             num_paths_onpol=10, 
             num_simulated_paths=1000,
             env_horizon=1000, 
             mpc_horizon=10,
             m_n_layers=2,
             m_size=500,
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
    # env = gym.make(env_name)
    env = HalfCheetahEnvNew()
    cost_fn = cheetah_cost_fn
    activation=tf.nn.relu
    output_activation=None

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    # max_path_length = max_path_length or env.spec.max_episode_steps
    max_path_length = max_path_length

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # Print environment infomation
    print("-------- env info --------")
    print("Environment name: ", env_name)
    print("Action space is discrete: ", discrete)
    print("Action space dim: ", ac_dim)
    print("Observation space dim: ", ob_dim)
    print("Max_path_length ", max_path_length)




    #========================================================================================#
    # Random data collection
    #========================================================================================#

    random_controller = RandomController(env)
    data_buffer_model = DataBuffer()
    data_buffer_ppo = DataBuffer_general(10000, 4)

    # sample path
    print("collecting random data .....  ")
    paths = sample(env, 
               random_controller, 
               num_paths=num_paths_random, 
               horizon=env_horizon, 
               render=False,
               verbose=False)

    # add into buffer
    for path in paths:
        for n in range(len(path['observations'])):
            data_buffer_model.add(path['observations'][n], path['actions'][n], path['next_observations'][n])

    print("data buffer size: ", data_buffer_model.size)

    normalization = compute_normalization(data_buffer_model)

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#
    tf_config = tf.ConfigProto() 
    tf_config.allow_soft_placement = True
    tf_config.intra_op_parallelism_threads =4
    tf_config.inter_op_parallelism_threads = 1
    sess = tf.Session(config=tf_config)

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    policy_nn = policy_network_ppo(sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate)

    if nn_baseline:
        value_nn = value_network(sess, ob_dim, n_layers, size, learning_rate)

    sess.__enter__() # equivalent to `with sess:`

    tf.global_variables_initializer().run()


    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        if MPC:
            dyn_model.fit(data_buffer_model)
        returns = []
        costs = []

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []

        while True:
            # print("data buffer size: ", data_buffer_model.size)
            current_path = {'observations': [], 'actions': [], 'reward': [], 'next_observations':[]}

            ob = env.reset()
            obs, acs, mpc_acs, rewards = [], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            return_ = 0
 
            while True:
                # print("steps ", steps)
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)

                if MPC:
                    mpc_ac = mpc_controller.get_action(ob)
                else:
                    mpc_ac = random_controller.get_action(ob)

                ac = policy_nn.predict(ob, mpc_ac)

                ac = ac[0]

                if not PG:
                    ac = mpc_ac

                acs.append(ac)
                mpc_acs.append(mpc_ac)

                current_path['observations'].append(ob)

                ob, rew, done, _ = env.step(ac)

                current_path['reward'].append(rew)
                current_path['actions'].append(ac)
                current_path['next_observations'].append(ob)

                return_ += rew
                rewards.append(rew)

                steps += 1
                if done or steps > max_path_length:
                    break


            if MPC:
                # cost & return
                cost = path_cost(cost_fn, current_path)
                costs.append(cost)
                returns.append(return_)
                print("total return: ", return_)
                print("costs: ", cost)

                # add into buffers
                for n in range(len(current_path['observations'])):
                    data_buffer_model.add(current_path['observations'][n], current_path['actions'][n], current_path['next_observations'][n])

            for n in range(len(current_path['observations'])):
                data_buffer_ppo.add(current_path['observations'][n], current_path['actions'][n], current_path['reward'][n], current_path['next_observations'][n])
        
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs),
                    "mpc_action" : np.array(mpc_acs)}



            paths.append(path)
            timesteps_this_batch += pathlength(path)
            # print("timesteps_this_batch", timesteps_this_batch)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch


        print("data_buffer_ppo.size:", data_buffer_ppo.size)


        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        mpc_ac_na = np.concatenate([path["mpc_action"] for path in paths])


        # Computing Q-values
     
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


        # Computing Baselines
        if nn_baseline:

            # b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no :ob_no})
            b_n = value_nn.predict(ob_no)
            b_n = normalize(b_n)
            b_n = denormalize(b_n, np.std(q_n), np.mean(q_n))
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # Advantage Normalization
        if normalize_advantages:
            adv_n = normalize(adv_n)

        # Optimizing Neural Network Baseline
        if nn_baseline:
            b_n_target = normalize(q_n)
            value_nn.fit(ob_no, b_n_target)
                # sess.run(baseline_update_op, feed_dict={sy_ob_no :ob_no, sy_baseline_target_n:b_n_target})


        # Performing the Policy Update

        # policy_nn.fit(ob_no, ac_na, adv_n)
        policy_nn.fit(ob_no, ac_na, adv_n, mpc_ac_na)

        # sess.run(update_op, feed_dict={sy_ob_no :ob_no, sy_ac_na:ac_na, sy_adv_n:adv_n})

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
    parser.add_argument('--env_name', type=str, default='HalfCheetah')
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
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
