from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

from baselines.common import Dataset, explained_variance, fmt_row, zipsame

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, sess, env, hid_size, num_hid_layers, clip_param, entcoeff, adam_epsilon,  gaussian_fixed_var=True):
        self.sess = sess
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers 

        self.pdtype = pdtype = make_pdtype(self.ac_space)
        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(self.ob_space.shape))
        self.ac = self.pdtype.sample_placeholder([None])

        with tf.variable_scope('pi'):
            self.ob_rms, self.vpred, self.pd, self._act = self.build_network(sess, 'pi', self.ob)
            self.pi_scope = tf.get_variable_scope().name

        with tf.variable_scope('old_pi'):
            _, _, self.old_pd, _ = self.build_network(sess, 'old_pi', self.ob)
            self.old_pi_scope = tf.get_variable_scope().name

        # Setup losses and stuff

        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

        self.adam, self.compute_losses, self.assign_old_eq_new, self.loss_names, self.lossandgrad = self.setup_ppo_loss(clip_param, entcoeff, adam_epsilon)


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

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, pd.sample(), pd.mode())
        _act = U.function([stochastic, ob], [ac, vpred])

        return ob_rms, vpred, pd, _act

    def setup_ppo_loss(self, clip_param, entcoeff, adam_epsilon):

        # Setup losses and stuff
        # ----------------------------------------

        clip_param = clip_param * self.lrmult # Annealed cliping parameter epislon

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
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = self.get_trainable_variables()
        lossandgrad = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], losses + [U.flatgrad(total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.get_old_variables(), self.get_variables())])
        compute_losses = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], losses)

        return adam, compute_losses, assign_old_eq_new, loss_names, lossandgrad

    def act_old(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def act(self, stochastic, ob):
        
        ac1, vpred1 =  self.sess.run([self.pd.sample(), self.vpred], feed_dict={self.ob: ob[None]})

        return ac1[0], vpred1[0]

    def get_old_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.old_pi_scope)
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.pi_scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.pi_scope)


    def get_initial_state(self):
        return []

