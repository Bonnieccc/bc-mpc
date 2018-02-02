import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController, MPCcontroller_BC_PPO, MPCcontroller_BC
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew
from data_buffer import DataBuffer, DataBuffer_general 
from behavioral_cloning import BCnetwork
# from pympler.tracker import SummaryTracker
from utils import denormalize, normalize, pathlength, sample, compute_normalization, path_cost
from ppo_bc_policy import MlpPolicy_bc



from mpi4py import MPI
from collections import deque
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger



# ===========================
# Training parameters for bc
# ===========================
BEHAVIORAL_CLONING = False
PPO = False

TEST_EPOCH = 5000
BATCH_SIZE_BC = 128
BC_BUFFER_SIZE = 20000
LOAD_MODEL = False
CHECKPOINT_DIR = 'checkpoints_bcmpc_noisy/'
############################

def behavioral_cloning(sess, env, bc_network, mpc_controller, env_horizon, bc_data_buffer, Training_epoch=1000):

    DAGGER = True
    # Imitation policy
    print("Behavioral cloning ..... ", " bc buffer size: ", bc_data_buffer.size)
    path = {'observations': [], 'actions': []}
    bc_network.train(bc_data_buffer, steps=1)

    for EP in range(Training_epoch):
        loss = bc_network.train(bc_data_buffer, steps=1)

        if EP % 500 == 0:
            print('epcho: ', EP, ' loss: ', loss)
            behavioral_cloning_eval(sess, env, bc_network, env_horizon)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def traj_segment_generator_old(pi, mpc_controller, mpc_controller_bc_ppo, bc_data_buffer, env, horizon):
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
    nxt_obs = np.array([ob for _ in range(horizon)])

    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac

        # _, vpred = pi.act(ob)

        # ac, vpred = pi.act(ob)

        vpred = 0.

        # if bc_data_buffer.size > 2000:
        #     ac = mpc_controller_bc_ppo.get_action(ob)
        # else:
        ac = mpc_controller.get_action(ob)


        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, done, _ = env.step(ac[0])

        nxt_obs[i] = ob
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1


        if t > 0 and t % horizon == 0:
            ob = env.reset()
            new = True 

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            new = False # marks if we're on first timestep of an episode


        if t > 0 and t % horizon == 0:
            print("ep_rets ", ep_rets)
            print("ep_lens ", ep_lens)
            ob = env.reset()
            cur_ep_ret = 0
            cur_ep_len = 0

            sec = copy.deepcopy({"ob" : obs, "rew" : rews, "nxt_ob": nxt_obs, "vpred" : vpreds, "new" : news,
            "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
            "ep_rets" : ep_rets, "ep_lens" : ep_lens})
            yield sec
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        t += 1

def traj_segment_generator(pi, mpc_controller, mpc_controller_bc_ppo, bc_data_buffer, env, horizon):
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

    while True:
        prevac = ac

        # _, vpred = pi.act(ob)

        # ac, vpred = pi.act(ob)

        vpred = 0.

        # if bc_data_buffer.size > 2000:
        #     ac = mpc_controller_bc_ppo.get_action(ob)
        # else:
        ac = mpc_controller.get_action(ob)

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, done, _ = env.step(ac)
        new = False

        nxt_obs[t] = ob
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1


        if t > 0 and t % (horizon-1) == 0:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)


        if t > 0 and t % (horizon-1) == 0:
            print("ep_rets ", ep_rets)
            print("ep_lens ", ep_lens)

            break

        t += 1

    sec = {"ob" : obs, "rew" : rews, "nxt_ob": nxt_obs, "vpred" : vpreds, "new" : news,
    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
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


def behavioral_cloning_eval(sess, env, bc_ppo_network, env_horizon):
    print('---------- BC PPO Performance ---------')
    st = env.reset_model()
    returns = 0

    for j in range(env_horizon):
        # at, vpred = bc_ppo_network.act(st)
        at = bc_ppo_network.predict(np.reshape(st, [1, -1]))[0]

        # print(at)
        nxt_st, r, _, _ = env.step(at)
        st = nxt_st
        returns += r

    print("return: ", returns)

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=10, 
         num_simulated_paths=10000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None,
         clip_param=0.2 , 
         entcoeff=0.0,
         gamma=0.99,
         lam=0.95,
         optim_epochs=10,
         optim_batchsize=64,
         schedule='linear',
         optim_stepsize=3e-4,
         timesteps_per_actorbatch=1000,
         bc_weight=0.5,
         ):

    start = time.time()

    logz.configure_output_dir(logdir)


    print("-------- env info --------")
    print("observation_space: ", env.observation_space.shape)
    print("action_space: ", env.action_space.shape)
    print(" ")


    random_controller = RandomController(env)
    model_data_buffer = DataBuffer()

    ppo_data_buffer = DataBuffer_general(BC_BUFFER_SIZE, 6)
    bc_data_buffer = DataBuffer_general(BC_BUFFER_SIZE, 2)

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
            model_data_buffer.add(path['observations'][n], path['actions'][n], path['next_observations'][n])


    print("model data buffer size: ", model_data_buffer.size)

    normalization = compute_normalization(model_data_buffer)

    #========================================================
    # 
    # Build dynamics model and MPC controllers and Behavioral cloning network.
    # 
    sess = tf.Session()

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

    policy_nn = MlpPolicy_bc(sess=sess, env=env, hid_size=64, num_hid_layers=2, clip_param=clip_param , entcoeff=entcoeff, bc_weight=bc_weight)
    # policy_nn = BCnetwork(sess, env, BATCH_SIZE_BC, learning_rate)

    # mpc_controller_bc_ppo = MPCcontroller_BC_PPO(env=env, 
    #                                dyn_model=dyn_model, 
    #                                bc_ppo_network=policy_nn,
    #                                horizon=mpc_horizon, 
    #                                cost_fn=cost_fn, 
    #                                num_simulated_paths=num_simulated_paths)
    mpc_controller_bc_ppo = MPCcontroller_BC(env=env, 
                                   dyn_model=dyn_model, 
                                   bc_network=policy_nn,
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    # init or load checkpoint with saver
    saver = tf.train.Saver()

    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    if checkpoint and checkpoint.model_checkpoint_path and LOAD_MODEL:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(CHECKPOINT_DIR):
          os.mkdir(CHECKPOINT_DIR)  

    #========================================================
    # 
    # Prepare for rollouts
    # 

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    max_timesteps = num_paths_onpol * env_horizon


    # for itr in range(1):
    #     """ YOUR CODE HERE """
    #     print("onpol_iters: ", itr)

    #     dyn_model.fit(model_data_buffer)


    #     returns = []
    #     costs = []

    #     for w in range(1):
    #         print("paths_onpol: ", w, " running.....")
    #         print("data buffer size: ", model_data_buffer.size)

    #         st = env.reset_model()
    #         path = {'observations': [], 'actions': [], 'next_observations':[]}
    #         # tracker.print_diff()

    #         return_ = 0

    #         for i in range(env_horizon):
    #             if render:
    #                 env.render()
    #             # print("env_horizon: ", i)   

    #             at = mpc_controller.get_action(st)
    #             # at = random_controller.get_action(st)

    #             st_next, env_reward, _, _ = env._step(at)
    #             path['observations'].append(st)
    #             path['actions'].append(at)
    #             path['next_observations'].append(st_next)
    #             st = st_next
    #             return_ += env_reward

    #         # cost & return
    #         cost = path_cost(cost_fn, path)
    #         costs.append(cost)
    #         returns.append(return_)
    #         print("total return: ", return_)
    #         print("costs: ", cost)



    for itr in range(onpol_iters):

        print("onpol_iters: ", itr)
        dyn_model.fit(model_data_buffer)

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)

        # saver.save(sess, CHECKPOINT_DIR)
        # behavioral_cloning_eval(sess, env, policy_nn, env_horizon)

        ppo_data_buffer.clear()
        seg = traj_segment_generator(policy_nn, mpc_controller, mpc_controller_bc_ppo, bc_data_buffer, env, env_horizon)
        add_vtarg_and_adv(seg, gamma, lam)


        ob, ac, rew, nxt_ob, atarg, tdlamret = seg["ob"], seg["ac"], seg["rew"], seg["nxt_ob"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        for n in range(len(ob)):
            ppo_data_buffer.add((ob[n], ac[n], rew[n], nxt_ob[n], atarg[n], tdlamret[n]))
            bc_data_buffer.add((ob[n], ac[n]))
            model_data_buffer.add(ob[n], ac[n], nxt_ob[n])

        print("ppo_data_buffer size", ppo_data_buffer.size)
        print("bc_data_buffer size", bc_data_buffer.size)
        print("model data buffer size: ", model_data_buffer.size)

        # optim_batchsize = optim_batchsize or ob.shape[0]

        # if hasattr(policy_nn, "ob_rms"): policy_nn.ob_rms.update(ob) # update running mean/std for policy
        # policy_nn.assign_old_eq_new() # set old parameter values to new parameter values
        
        behavioral_cloning(sess, env, policy_nn, mpc_controller, env_horizon, bc_data_buffer, Training_epoch=1000)

        # for op_ep in range(optim_epochs):
        #     # losses = [] # list of tuples, each of which gives the loss for a minibatch
        #     # for i in range(int(timesteps_per_actorbatch/optim_batchsize)):

        #     if PPO:
        #         sample_ob_no, sample_ac_na, sample_rew, sample_nxt_ob_no, sample_adv_n, sample_b_n_target = ppo_data_buffer.sample(optim_batchsize)
        #         newlosses = policy_nn.lossandupdate_ppo(sample_ob_no, sample_ac_na, sample_adv_n, sample_b_n_target, cur_lrmult, optim_stepsize*cur_lrmult)
        #         # losses.append(newlosses)

        #     if BEHAVIORAL_CLONING:
        #         sample_ob_no, sample_ac_na = bc_data_buffer.sample(optim_batchsize)
        #         policy_nn.update_bc(sample_ob_no, sample_ac_na, optim_stepsize*cur_lrmult)

        #     if op_ep % 500 == 0:
        #         print('epcho: ', op_ep)
        #         behavioral_cloning_eval(sess, env, policy_nn, env_horizon)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values


        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        ep_lengths = seg["ep_lens"]
        returns =  seg["ep_rets"]

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", iters_so_far)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", timesteps_so_far)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mpc_bc_ppo')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Model Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=100)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=260)
    parser.add_argument('--batch_size', '-b', type=int, default=512)

    # BC and PPO Training args
    parser.add_argument('--optim_stepsize', '-olr', type=float, default=3e-4)
    parser.add_argument('--clip_param', '-cp', type=float, default=0.2)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--entcoeff', '-ent', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--optim_epochs', type=int, default=1000)
    parser.add_argument('--optim_batchsize', type=int, default=128)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--timesteps_per_actorbatch', '-b2', type=int, default=1000)
    parser.add_argument('--bc_weight', '-bcw', type=float, default=0.5)

    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=1)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=10)
    args = parser.parse_args()


    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
        
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths, 
                 num_paths_onpol=args.onpol_paths, 
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len, 
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 clip_param = args.clip_param,
                 entcoeff = args.entcoeff,
                 gamma = args.gamma,
                 lam = args.lam,
                 optim_epochs = args.optim_epochs,
                 optim_batchsize = args.optim_batchsize,
                 schedule = args.schedule,
                 optim_stepsize = args.optim_stepsize,
                 timesteps_per_actorbatch = args.timesteps_per_actorbatch,
                 bc_weight=args.bc_weight,
                 )

if __name__ == "__main__":
    main()
