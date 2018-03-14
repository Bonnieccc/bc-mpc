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
tf.app.flags.DEFINE_string('CHECKPOINT_DIR', 'checkpoints_bcmpc_noisy/', "Checkpoints save directory")
tf.app.flags.DEFINE_boolean('LOAD_MODEL', False, """Load model or not""")
tf.app.flags.DEFINE_boolean('SELFEXP', False, """Use external exploration or ppo's own exp""")
tf.app.flags.DEFINE_float('MPC_EXP', 0.5, 'MPC external explore magnitude')


# Experiment meta-params
tf.app.flags.DEFINE_string('env_name', 'HalfCheetah-v1', 'Environment name')
tf.app.flags.DEFINE_string('exp_name', 'mpc_bc_ppo', 'Experiment name')
tf.app.flags.DEFINE_integer('seed', 3, 'random seed')
tf.app.flags.DEFINE_boolean('render', False, 'Render or not')

# Model Training args
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.app.flags.DEFINE_integer('onpol_iters', 100, 'onpol_iters')
tf.app.flags.DEFINE_integer('dyn_iters', 200, 'dyn_iters')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch_size')
tf.app.flags.DEFINE_integer('MODELBUFFER_SIZE', 1000000, 'MODELBUFFER_SIZE')

# BC and PPO Training args
tf.app.flags.DEFINE_float('bc_lr', 1e-3, '')
tf.app.flags.DEFINE_float('ppo_lr',  3e-4, '')
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
tf.app.flags.DEFINE_integer('mpc_horizon', 7, '')

tf.app.flags.DEFINE_boolean('mpc', False, 'Render or not')
tf.app.flags.DEFINE_boolean('bc', False, 'Render or not')
tf.app.flags.DEFINE_boolean('ppo', True, 'Render or not')

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

    logz.configure_output_dir(logdir)
    merged_summary, summary_writer, ppo_return_op, mpc_return_op, model_loss_op, ppo_std_op, mpc_std_op = build_summary_ops(logdir)

    print("-------- env info --------")
    print("Environment: ", FLAGS.env_name)
    print("observation_space: ", env.observation_space.shape)
    print("action_space: ", env.action_space.shape)
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

    checkpoint = tf.train.get_checkpoint_state(FLAGS.CHECKPOINT_DIR)

    if checkpoint and checkpoint.model_checkpoint_path and FLAGS.LOAD_MODEL:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(FLAGS.CHECKPOINT_DIR):
          os.mkdir(FLAGS.CHECKPOINT_DIR)  

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
    bc = False
    ppo_mpc = False
    mpc_returns = 0
    model_loss = 0
    for itr in range(onpol_iters):

        print(" ")

        print("onpol_iters: ", itr)

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            

        print("bc learning_rate: ",  bc_lr)
        print("ppo learning_rate: ",  ppo_lr)


        ################## fit mpc model
        if MPC:
            model_loss = dyn_model.fit(model_data_buffer)


        ################## ppo seg data
        ppo_data_buffer.clear()

        # ppo_seg = traj_segment_generator_ppo(policy_nn, env, env_horizon)
        mpc = False
        ppo_seg = traj_segment_generator(policy_nn, mpc_controller, mpc_ppo_controller, bc_data_buffer, env, mpc, ppo_mpc, env_horizon)

        add_vtarg_and_adv(ppo_seg, gamma, lam)

        ob, ac, rew, nxt_ob, atarg, tdlamret = \
        ppo_seg["ob"], ppo_seg["ac"], ppo_seg["rew"], ppo_seg["nxt_ob"], ppo_seg["adv"], ppo_seg["tdlamret"]

        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        # add into buffer
        for n in range(len(ob)):
            ppo_data_buffer.add([ob[n], ac[n], atarg[n], tdlamret[n]])
            model_data_buffer.add([ob[n], ac[n], rew[n], nxt_ob[n], nxt_ob[n]-ob[n]])

        ppo_std = np.std(ac, axis=0)
        print("ppo_std: ", ppo_std)

        ################## mpc augmented seg data

        if itr % FLAGS.MPC_AUG_GAP == 0 and MPC:
            print("MPC AUG PPO")

            ppo_mpc = True
            mpc = True
            mpc_seg = traj_segment_generator(policy_nn, mpc_controller, mpc_ppo_controller, bc_data_buffer, env, mpc, ppo_mpc, env_horizon)
            add_vtarg_and_adv(mpc_seg, gamma, lam)

            ob, ac, mpcac, rew, nxt_ob, atarg, tdlamret = mpc_seg["ob"], mpc_seg["ac"], mpc_seg["mpcac"], mpc_seg["rew"], mpc_seg["nxt_ob"], mpc_seg["adv"], mpc_seg["tdlamret"]
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            # # add into buffer
            # for n in range(len(ob)):
            #     # if PPO:
            #     #     ppo_data_buffer.add([ob[n], ac[n], atarg[n], tdlamret[n]])

            #     if BEHAVIORAL_CLONING and bc:
            #         bc_data_buffer.add([ob[n], mpcac[n]])

            #     if MPC:
            #         model_data_buffer.add([ob[n], mpcac[n], rew[n], nxt_ob[n], nxt_ob[n]-ob[n]])

            mpc_returns = mpc_seg["ep_rets"]
            mpc_std = np.std(mpcac)
            print("mpc_std: ", mpc_std)
        if not mpc:
            mpc_std = 0

        seg = ppo_seg

        # check if seg is good
        ep_lengths = seg["ep_lens"]
        returns =  seg["ep_rets"]

        # saver.save(sess, FLAGS.CHECKPOINT_DIR)
        if BEHAVIORAL_CLONING:
            if np.mean(returns) > 100:
                bc = True
            else:
                bc = False

            print("BEHAVIORAL_CLONING: ", bc)


            bc_return = policy_net_eval(sess, env, policy_nn, env_horizon)

            if bc_return > 100:
                ppo_mpc = True
            else:
                ppo_mpc = False


        ################## optimization

        print("ppo_data_buffer size", ppo_data_buffer.size)
        print("bc_data_buffer size", bc_data_buffer.size)
        print("model data buffer size: ", model_data_buffer.size)

        # optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(policy_nn, "ob_rms"): policy_nn.ob_rms.update(ob) # update running mean/std for policy
        policy_nn.assign_old_eq_new() # set old parameter values to new parameter values
        
        for op_ep in range(optim_epochs):
            # losses = [] # list of tuples, each of which gives the loss for a minibatch
            # for i in range(int(timesteps_per_actorbatch/optim_batchsize)):

            if PPO:
                sample_ob_no, sample_ac_na, sample_adv_n, sample_b_n_target = ppo_data_buffer.sample(optim_batchsize)
                newlosses = policy_nn.lossandupdate_ppo(sample_ob_no, sample_ac_na, sample_adv_n, sample_b_n_target, cur_lrmult, ppo_lr*cur_lrmult)
                # losses.append(newlosses)

            if BEHAVIORAL_CLONING and bc:
                sample_ob_no, sample_ac_na = bc_data_buffer.sample(optim_batchsize)
                # print("sample_ob_no", sample_ob_no.shape)
                # print("sample_ac_na", sample_ac_na.shape)

                policy_nn.update_bc(sample_ob_no, sample_ac_na, bc_lr*cur_lrmult)

            if op_ep % (100) == 0 and BEHAVIORAL_CLONING and bc:
                print('epcho: ', op_ep)
                policy_net_eval(sess, env, policy_nn, env_horizon)


        ################## print and save data

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values


        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1        


        logz.log_tabular("TimeSoFar", time.time() - start)
        logz.log_tabular("TimeEp", time.time() - tstart)
        logz.log_tabular("Iteration", iters_so_far)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("MpcReturn", np.mean(mpc_returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", timesteps_so_far)
        logz.dump_tabular()
        logz.pickle_tf_vars()
        tstart = time.time()

        ################### TF Summaries
        summary_str = sess.run(merged_summary, feed_dict={
                  ppo_return_op:np.mean(returns),
                  mpc_return_op:np.mean(mpc_returns),
                  model_loss_op:model_loss,
                  ppo_std_op:ppo_std,
                  mpc_std_op:mpc_std,
                  })
        summary_writer.add_summary(summary_str, itr)
        summary_writer.flush()

def build_summary_ops(logdir):

    summary_writer = tf.summary.FileWriter(logdir)

    ppo_return_op =  tf.placeholder(tf.float32)
    mpc_return_op =  tf.placeholder(tf.float32)
    model_loss_op = tf.placeholder(tf.float32)
    ppo_std_op =  tf.placeholder(tf.float32, shape=(6),)
    mpc_std_op = tf.placeholder(tf.float32)

    tf.summary.scalar('mean_ppo_return', ppo_return_op)
    tf.summary.scalar('mean_mpc_return', mpc_return_op)
    tf.summary.scalar('model_loss', model_loss_op)

    tf.summary.histogram('ppo_action_std', ppo_std_op)
    tf.summary.scalar('mpc_action_std', mpc_std_op)

    merged_summary = tf.summary.merge_all()

    return merged_summary, summary_writer, ppo_return_op, mpc_return_op, model_loss_op, ppo_std_op, mpc_std_op





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
        env.seed(FLAGS.seed)
        cost_fn = cheetah_cost_fn
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
