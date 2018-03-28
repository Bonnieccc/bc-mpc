import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsRewardModel, NNDynamicsModel
from controllers import MPCcontroller, RandomController, MPCcontrollerPolicyNet, MPCcontrollerPolicyNetReward
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew
from data_buffer import DataBuffer, DataBufferGeneral 
from behavioral_cloning import BCnetwork
from utils import denormalize, normalize, pathlength, sample, compute_normalization, path_cost, flatten_lists, traj_segment_generator, add_vtarg_and_adv, policy_net_eval
from ppo_bc_policy import MlpPolicy

from mpi4py import MPI
from collections import deque
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger


# ===========================
# TF Flags
# ===========================

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('LEARN_REWARD', False, "Learn reward function or use cost function as mpc evaluation")
tf.app.flags.DEFINE_integer('MPC_AUG_GAP', 1, "How many iters to use mpc augumentation ")
tf.app.flags.DEFINE_integer('SAVE_ITER', 1, "In how many iterations save model once")
tf.app.flags.DEFINE_boolean('LOAD_MODEL', True, """Load model or not""")
tf.app.flags.DEFINE_boolean('SELFEXP', True, """Use external exploration or ppo's own exp""")
tf.app.flags.DEFINE_float('MPC_EXP', 0.5, 'MPC external explore ratio [0, 1]')


# Experiment meta-params
tf.app.flags.DEFINE_string('env_name', 'HalfCheetah-v1', 'Environment name')
tf.app.flags.DEFINE_string('exp_name', 'temp', 'Experiment name')
tf.app.flags.DEFINE_string('model_path', 'temp', 'model')

tf.app.flags.DEFINE_integer('seed', 3, 'random seed')
tf.app.flags.DEFINE_boolean('render', False, 'Render or not')

# Model Training args
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.app.flags.DEFINE_integer('onpol_iters', 100, 'onpol_iters')
tf.app.flags.DEFINE_integer('dyn_iters', 200, 'dyn_iters')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch_size')
tf.app.flags.DEFINE_integer('MODELBUFFER_SIZE', 1000000, 'MODELBUFFER_SIZE')
tf.app.flags.DEFINE_boolean('LAYER_NORM', True, """Use layer normalization""")

# BC and PPO Training args
tf.app.flags.DEFINE_float('bc_lr', 1e-3, '')
tf.app.flags.DEFINE_float('ppo_lr',  1e-4, '')
tf.app.flags.DEFINE_float('clip_param', 0.2, '')
tf.app.flags.DEFINE_float('gamma', 0.99, '')
tf.app.flags.DEFINE_float('entcoeff',  0.0, '')
tf.app.flags.DEFINE_float('lam', 0.95, '')

tf.app.flags.DEFINE_integer('optim_epochs', 500, '')
tf.app.flags.DEFINE_integer('optim_batchsize', 128, '')
tf.app.flags.DEFINE_integer('timesteps_per_actorbatch', 1000, '')

tf.app.flags.DEFINE_string('schedule', 'constant', '')

# Data collection
tf.app.flags.DEFINE_integer('random_paths', 10, '')
tf.app.flags.DEFINE_integer('onpol_paths', 1, '')
tf.app.flags.DEFINE_integer('simulated_paths', 400, '')
tf.app.flags.DEFINE_integer('ep_len', 1000, '')
# Neural network architecture args
tf.app.flags.DEFINE_integer('n_layers', 2, '')
tf.app.flags.DEFINE_integer('size', 256, '')
# MPC Controller
tf.app.flags.DEFINE_integer('mpc_horizon', 30, '')

tf.app.flags.DEFINE_boolean('mpc', False, 'mpc or not')
tf.app.flags.DEFINE_boolean('mpc_rand', False, 'mpc_rand or not')
tf.app.flags.DEFINE_boolean('ppo', True, 'ppo or not')
tf.app.flags.DEFINE_boolean('bc', False, 'bc or not')

############################
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
         bc_lr=1e-3,
         ppo_lr=3e-4,
         timesteps_per_actorbatch=1000,
         MPC = True,
         BEHAVIORAL_CLONING = True,
         PPO = True,
         ):

    start = time.time()


    print("-------- env info --------")
    print("Environment: ", FLAGS.env_name)
    print("observation_space: ", env.observation_space.shape)
    print("action_space: ", env.action_space.shape)
    print("action_space low: ", env.action_space.low)
    print("action_space high: ", env.action_space.high)

    print("BEHAVIORAL_CLONING: ", BEHAVIORAL_CLONING)
    print("PPO: ", PPO)
    print("MPC-AUG: ", MPC)

    print(" ")


    random_controller = RandomController(env)

    # Creat buffers
    model_data_buffer = DataBufferGeneral(FLAGS.MODELBUFFER_SIZE, 5)
    ppo_data_buffer = DataBufferGeneral(10000, 4)
    bc_data_buffer = DataBufferGeneral(2000, 2)

    # Random sample path

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
            model_data_buffer.add([path['observations'][n],
                                 path['actions'][n], 
                                 path['rewards'][n], 
                                 path['next_observations'][n], 
                                 path['next_observations'][n] - path['observations'][n]])

    print("model data buffer size: ", model_data_buffer.size)

    normalization = compute_normalization(model_data_buffer)

    #========================================================
    # 
    # Build dynamics model and MPC controllers and Behavioral cloning network.
    # 
    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    tf_config = tf.ConfigProto() 

    tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)

    policy_nn = MlpPolicy(sess=sess, env=env, hid_size=128, num_hid_layers=2, clip_param=clip_param , entcoeff=entcoeff)

    if FLAGS.LEARN_REWARD:
        print("Learn reward function")
        dyn_model = NNDynamicsRewardModel(env=env, 
                                        normalization=normalization,
                                        batch_size=batch_size,
                                        iterations=dynamics_iters,
                                        learning_rate=learning_rate,
                                        sess=sess)

        mpc_ppo_controller = MPCcontrollerPolicyNetReward(env=env, 
                                       dyn_model=dyn_model, 
                                       explore=FLAGS.MPC_EXP,
                                       policy_net=policy_nn,
                                       self_exp=FLAGS.SELFEXP,
                                       horizon=mpc_horizon, 
                                       num_simulated_paths=num_simulated_paths)
    else:
        print("Use predefined cost function")
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

        mpc_ppo_controller = MPCcontrollerPolicyNet(env=env, 
                                       dyn_model=dyn_model, 
                                       explore=FLAGS.MPC_EXP,
                                       policy_net=policy_nn,
                                       self_exp=FLAGS.SELFEXP,
                                       horizon=mpc_horizon, 
                                       cost_fn=cost_fn, 
                                       num_simulated_paths=num_simulated_paths)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)
    # if not PPO:
    #     mpc_ppo_controller = mpc_controller

    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    # init or load checkpoint with saver
    saver = tf.train.Saver()

    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_path)

    print("checkpoint", checkpoint)

    if checkpoint and checkpoint.model_checkpoint_path and FLAGS.LOAD_MODEL:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(FLAGS.model_path):
          os.mkdir(FLAGS.model_path)  

    #========================================================
    # 
    # Prepare for rollouts
    # 

    tstart = time.time()


    states_true = []
    states_predict = []
    rewards_true = []
    rewards_predict = []
    ob = env.reset()
    ob_pre = np.expand_dims(ob, axis=0)

    states_true.append(ob)
    states_predict.append(ob_pre)

    for step in range(100):
        # ac = env.action_space.sample() # not used, just so we have the datatype
        ac, _ = policy_nn.act(ob, stochastic=True)
        ob, rew, done, _ = env.step(ac)
        ob_pre, r_pre = dyn_model.predict(ob_pre, ac)
        states_true.append(ob)
        rewards_true.append(rew)
        states_predict.append(ob_pre)
        rewards_predict.append(r_pre[0][0])

    states_true = np.asarray(states_true)
    states_predict = np.asarray(states_predict)
    states_predict = np.squeeze(states_predict, axis=1)
    rewards_true = np.asarray(rewards_true)
    rewards_predict = np.asarray(rewards_predict)

    print("states_true", states_true.shape)
    print("states_predict", states_predict.shape)
    print("rewards_true", rewards_true.shape)
    print("rewards_predict", rewards_predict.shape)

    np.savetxt('./data/eval_model/states_true.out', states_true, delimiter=',') 
    np.savetxt('./data/eval_model/states_predict.out', states_predict, delimiter=',') 

    np.savetxt('./data/eval_model/rewards_true.out', rewards_true, delimiter=',') 
    np.savetxt('./data/eval_model/rewards_predict.out', rewards_predict, delimiter=',') 

    # ################# PPO deterministic evaluation
    # ppo_determinisitc_return = policy_net_eval(sess, env, policy_nn, env_horizon, stochastic=False)





def main():

    assert (FLAGS.mpc or FLAGS.ppo) == True

    # Set seed
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    # logdir = FLAGS.exp_name + '_' + FLAGS.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = FLAGS.exp_name + '_' + FLAGS.env_name

    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if FLAGS.env_name == "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn

        # env = gym.make(FLAGS.env_name)
        env.seed(FLAGS.seed)
    else:
        env = gym.make(FLAGS.env_name)
        env.seed(FLAGS.seed)
        cost_fn = None
        
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=FLAGS.render,
                 learning_rate=FLAGS.learning_rate,
                 onpol_iters=FLAGS.onpol_iters,
                 dynamics_iters=FLAGS.dyn_iters,
                 batch_size=FLAGS.batch_size,
                 num_paths_random=FLAGS.random_paths, 
                 num_paths_onpol=FLAGS.onpol_paths, 
                 num_simulated_paths=FLAGS.simulated_paths,
                 env_horizon=FLAGS.ep_len, 
                 mpc_horizon=FLAGS.mpc_horizon,
                 n_layers = FLAGS.n_layers,
                 size=FLAGS.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 clip_param = FLAGS.clip_param,
                 entcoeff = FLAGS.entcoeff,
                 gamma = FLAGS.gamma,
                 lam = FLAGS.lam,
                 optim_epochs = FLAGS.optim_epochs,
                 optim_batchsize = FLAGS.optim_batchsize,
                 schedule = FLAGS.schedule,
                 bc_lr = FLAGS.bc_lr,
                 ppo_lr = FLAGS.ppo_lr,
                 timesteps_per_actorbatch = FLAGS.timesteps_per_actorbatch,
                 MPC = FLAGS.mpc,
                 BEHAVIORAL_CLONING = FLAGS.bc,
                 PPO = FLAGS.ppo,
                 )

if __name__ == "__main__":
    main()
