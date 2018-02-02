from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines.common.distributions import make_pdtype

import baselines.common.tf_util as U
import tensorflow as tf
import gym

# from baselines.common.mpi_adam import MpiAdam
# from baselines.common.mpi_moments import mpi_moments



class MlpPolicy_bc(object):
    recurrent = False

    def __init__(self, sess, env, hid_size, num_hid_layers, clip_param, entcoeff, bc_weight=0.5):
        self.sess = sess
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers 
        self.bc_weight = bc_weight
        self.pdtype = pdtype = make_pdtype(self.ac_space)

        self.ob = tf.placeholder(name="ob", dtype=tf.float32, shape=[None] + list(self.ob_space.shape))
        self.ac = self.pdtype.sample_placeholder([None])

        with tf.variable_scope('pi'):
            self.ob_rms, self.vpred, self.pd, self.sample_ac, self.ac_mean = self.build_network(sess, 'pi', self.ob)
            self.pi_scope = tf.get_variable_scope().name

        with tf.variable_scope('old_pi'):
            _, _, self.old_pd, _, _ = self.build_network(sess, 'old_pi', self.ob)
            self.old_pi_scope = tf.get_variable_scope().name

        # Setup losses and stuff

        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

        self.ppo_loss, self.bc_loss, self.loss_names, self.losses, self.var_list = self.setup_ppo_loss(clip_param, entcoeff)

        self.gradients = tf.gradients(self.ppo_loss, self.var_list)
        self.assign_old_eq_new_op = [tf.assign(oldv, newv) for (oldv, newv) in zipsame(self.get_old_variables(), self.get_variables())]

        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.update_op_ppo = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ppo_loss, var_list=self.var_list)
        self.update_op_bc = tf.train.AdamOptimizer(self.learning_rate).minimize(self.bc_loss, var_list=self.var_list)

    def build_network(self, sess, scope, ob):

        with tf.variable_scope(scope + "/obfilter"):
            ob_rms = RunningMeanStd(shape=self.ob_space.shape)

        with tf.variable_scope(scope + '/vf'):
            obz = tf.clip_by_value((ob - ob_rms.mean) / ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, self.hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope(scope + '/pol'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, self.hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            mean = tf.layers.dense(last_out, self.pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)


        pd = self.pdtype.pdfromflat(pdparam)

        sample_ac = pd.sample()
        ac_mean = pd.mode()

        return ob_rms, vpred, pd, sample_ac, ac_mean

    def setup_ppo_loss(self, clip_param, entcoeff):

        # Setup losses and stuff
        # ----------------------------------------

        clip_param = clip_param * self.lrmult # Annealed cliping parameter epislon

        # ppo loss
        kloldnew = self.old_pd.kl(self.pd)
        ent = self.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(self.pd.logp(self.ac) - self.old_pd.logp(self.ac)) # pnew / pold
        surr1 = ratio * self.atarg # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * self.atarg #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.reduce_mean(tf.square(self.vpred - self.ret))


        # bc loss
        bc_loss = tf.reduce_mean(tf.square(tf.subtract(self.ac, self.ac_mean)))

        ppo_loss = pol_surr + pol_entpen + vf_loss

        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, bc_loss]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "bc_loss"]


        var_list = var_list = self.get_trainable_variables()

        return ppo_loss, bc_loss, loss_names, losses, var_list

    def compute_losses(self, ob, ac, atarg, ret, cur_lrmult):
        return self.sess.run([self.losses],
                 feed_dict={self.ob: ob,
                            self.ac: ac, 
                            self.atarg: atarg,
                            self.ret: ret,
                            self.lrmult:cur_lrmult, 
                            })

    def lossandupdate_ppo(self, ob, ac, atarg, ret, cur_lrmult, learning_rate):
         losses, _ = self.sess.run([self.losses, self.update_op_ppo],
                 feed_dict={self.ob: ob,
                            self.ac: ac, 
                            self.atarg: atarg,
                            self.ret: ret,
                            self.lrmult:cur_lrmult, 
                            self.learning_rate:learning_rate, 
                            })
         return losses

    def update_bc(self, ob, ac, learning_rate):
        self.sess.run(self.update_op_bc,
                 feed_dict={self.ob: ob,
                            self.ac: ac, 
                            self.learning_rate:learning_rate, 
                            })

    def get_grad(self, ob, ac, atarg, ret, cur_lrmult):
        return self.sess.run([self.gradients],
                 feed_dict={self.ob: ob,
                            self.ac: ac, 
                            self.atarg: atarg,
                            self.ret: ret,
                            self.lrmult:cur_lrmult, 
                            })

    def optimize(self, lr, ob, ac, atarg, ret):
        self.sess.run(self.update_op_ppo, feed_dict={self.learning_rate:lr,
                                                 self.ob: ob,
                                                 self.ac: ac, 
                                                 self.atarg: atarg,
                                                 self.ret: ret})


    def assign_old_eq_new(self):
        self.sess.run([self.assign_old_eq_new_op])


    def act(self, ob):
        
        if ob.ndim == 1:
        # ensure dim is [?, s_dim]
            ob = ob[None]

        ac1, vpred1 =  self.sess.run([self.sample_ac, self.vpred], feed_dict={self.ob: ob})

        return ac1, vpred1

    def get_old_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.old_pi_scope)
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.pi_scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.pi_scope)


    def get_initial_state(self):
        return []

