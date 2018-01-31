import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U

from distributions import make_pdtype

from utils import build_mlp


class policy_network_mpc(object):
    """docstring for policy_network"""
    def __init__(self, sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate):
        self.sess = sess
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.size = size 

        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_sampled_ac, self.sy_logprob_n = self.build_policy_network()

        # setup loss
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build_policy_network(self):
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 

        # Define a placeholder for advantages
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)


        if self.discrete:
            # YOUR_CODE_HERE
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, 'policy_network', n_layers=self.n_layers, size=self.size)
            sy_prob_na = tf.nn.softmax(sy_logits_na)
            # print('sy_logits_na', sy_logits_na)

            sy_sampled_ac = tf.multinomial(sy_logits_na, 1) # Hint: Use the tf.multinomial op
            sy_sampled_ac = tf.reshape(sy_sampled_ac, [-1])
            # print('sy_sampled_ac', sy_sampled_ac)

            sy_ac_onehot_na = tf.one_hot(sy_ac_na, depth=self.ac_dim)
            # print('sy_ac_onehot_na', sy_ac_onehot_na)
            sy_ac_responseprob_na = tf.reduce_mean(tf.multiply(sy_ac_onehot_na,  sy_prob_na), axis=1)
            # print('sy_ac_responseprob_na', sy_ac_responseprob_na)
            sy_logprob_n = tf.log(sy_ac_responseprob_na)
            # print('sy_logprob_n', sy_logprob_n)

        else:
            # YOUR_CODE_HERE
            # Parameterize the gaussian policy by mean and std
            self.mpc_a = tf.placeholder(shape=[None, self.ac_dim], name="mpc_a", dtype=tf.float32) 

            net = sy_ob_no


            net = tf.layers.dense(net, 64, activation=tf.tanh)

            net = tf.layers.dense(net, 64, activation=tf.tanh)
  
            net = tf.concat([net, self.mpc_a], axis=1)

            sy_mean_na = tf.layers.dense(net, self.ac_dim, activation=None)

            # gate = tf.Variable(tf.ones([1, self.ac_dim]))

            # sy_mean_na = gate *  self.mpc_a  + (1-gate) * sy_mean_na

            # sy_mean_na = self.mpc_a

            self.sy_logstd = tf.Variable(tf.zeros([1, self.ac_dim]) , name='action/logstd', dtype=tf.float32) # logstd should just be a trainable variable, not a network output.
            self.sy_std = tf.exp(self.sy_logstd)
            # print('self.sy_std', self.sy_std)

            # Sample an action
            sy_sampled_ac = sy_mean_na + self.sy_std * tf.random_normal(tf.shape(sy_mean_na))
            # print('sy_sampled_ac', sy_sampled_ac)

            # Log likely hood of chosen this action
            # Hint: Use the log probability under a multivariate gaussian.
            sy_z = (sy_ac_na - sy_mean_na)/self.sy_std
            # print('sy_z', sy_z)
            sy_logprob_n = -0.5 * tf.square(sy_z) - 0.5 * tf.log(tf.constant(2*np.pi)) - self.sy_logstd

            sy_logprob_n = tf.reduce_sum(sy_logprob_n, axis=1)
            # print('sy_logprob_n', sy_logprob_n)

        return sy_ob_no, sy_ac_na, sy_adv_n, sy_sampled_ac, sy_logprob_n

    def predict(self, ob, mpc_a):
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None], self.mpc_a: mpc_a[None]})

    def fit(self, ob_no, ac_na, adv_n, mpc_a):
        self.sess.run(self.update_op, feed_dict={self.sy_ob_no :ob_no, self.sy_ac_na:ac_na, self.sy_adv_n:adv_n, self.mpc_a: mpc_a})

    # def predict(self, ob):
    #     return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})

    # def fit(self, ob_no, ac_na, adv_n):
    #     self.sess.run(self.update_op, feed_dict={self.sy_ob_no :ob_no, self.sy_ac_na:ac_na, self.sy_adv_n:adv_n})


class policy_network(object):
    """docstring for policy_network"""
    def __init__(self, sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate):
        self.sess = sess
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.size = size 

        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_sampled_ac, self.sy_logprob_n = self.build_policy_network()

        # setup loss
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_policy_network(self):
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 

        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        sy_logstd = tf.Variable(tf.zeros([self.ac_dim]), name='action/logstd', dtype=tf.float32) # logstd should just be a trainable variable, not a network output.

        net = sy_ob_no

        net = tf.layers.dense(net, 64, activation=tf.tanh)

        net = tf.layers.dense(net, 64, activation=tf.tanh)


        sy_mean_na = tf.layers.dense(net, self.ac_dim, activation=None)

        sy_std = tf.exp(sy_logstd)
        # print('sy_std', sy_std)

        # Sample an action
        sy_sampled_ac = sy_mean_na + sy_std * tf.random_normal(tf.shape(sy_mean_na))
        # print('sy_sampled_ac', sy_sampled_ac)

        # Log likely hood of chosen this action
        # Hint: Use the log probability under a multivariate gaussian.
        sy_z = (sy_ac_na - sy_mean_na)/sy_std
        # print('sy_z', sy_z)
        sy_logprob_n = -0.5 * tf.square(sy_z) - 0.5 * tf.log(tf.constant(2*np.pi)) - sy_logstd

        sy_logprob_n = tf.reduce_sum(sy_logprob_n, axis=1)
        # print('sy_logprob_n', sy_logprob_n)

        return sy_ob_no, sy_ac_na, sy_adv_n, sy_sampled_ac, sy_logprob_n

    def predict(self, ob):
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})

    def fit(self, ob_no, ac_na, adv_n):
        self.sess.run(
            self.update_op,
            feed_dict={self.sy_ob_no :ob_no, 
                       self.sy_ac_na:ac_na, 
                       self.sy_adv_n:adv_n})


class policy_network_ppo(object):
    """docstring for policy_network"""
    def __init__(self, sess, ob_dim, ac_dim, discrete, n_layers, size, learning_rate):
        self.sess = sess
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.size = size 


        # PPO hyper-parameters
        self.clip_param = 0.2 
        self.entcoeff = 0.0


        # input placeholder
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.sy_baseline_target_n = tf.placeholder(shape=[None], name="baseline_target", dtype=tf.float32)

        # build policy network
        self.pd = self.build_policy_network(self.sy_ob_no)
        self.baseline_prediction = self.build_value_network(self.sy_ob_no, scope='value')

        self.network_params = tf.trainable_variables()

        self.pd_old = self.build_policy_network(self.sy_ob_no)
        _ = self.build_value_network(self.sy_ob_no, scope='old_value')

        # output
        self.sy_sampled_ac = self.pd.sample()

        # Op for sync old network with online network weights
        self.old_network_params = tf.trainable_variables()[len(self.network_params):]
        self.sync_old_network_params = [self.old_network_params[i].assign(self.network_params[i])  for i in range(len(self.old_network_params))]

        # self.setup_vpg_loss(self.sy_ac_na, self.sy_adv_n)
        self.setup_ppo_loss(self.sy_ac_na, self.sy_adv_n)

        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.network_params)

    def setup_ppo_loss(self, ac, adv):
        kloldnew = self.pd_old.kl(self.pd)
        ent = self.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-self.entcoeff) * meanent
        value_loss = tf.reduce_mean(tf.square(self.baseline_prediction - self.sy_baseline_target_n))


        ratio = tf.exp(self.pd.logp(ac) - self.pd_old.logp(ac)) # pnew / pold
        surr1 = ratio * adv # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv 
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        self.loss = pol_surr + pol_entpen + value_loss


    def setup_vpg_loss(self, ac, adv):
        value_loss = tf.reduce_mean(tf.square(self.baseline_prediction - self.sy_baseline_target_n))

        # setup loss
        sy_logprob_n = self.pd.logp(ac)
        self.loss = -tf.reduce_mean(sy_logprob_n * adv) # Loss function that we'll differentiate to get the policy gradient.
        self.loss += value_loss

    def build_policy_network(self, sy_ob_no):


        net = sy_ob_no
        net = tf.layers.dense(net, 64, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=1.0))
        net = tf.layers.dense(net, 64, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=1.0))
        sy_mean_na = tf.layers.dense(net, self.ac_dim, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        sy_logstd = tf.Variable(tf.zeros([self.ac_dim]), name='action/logstd', dtype=tf.float32) # logstd should just be a trainable variable, not a network output.
        # construct distribution
        pdparam = tf.concat([sy_mean_na, sy_mean_na * 0.0 + sy_logstd], axis=1)
        pdtype = make_pdtype(self.ac_dim)
        pd = pdtype.pdfromflat(pdparam)

        return pd

    def build_value_network(self, sy_ob_no, scope):

        net = tf.nn.tanh(tf.layers.dense(sy_ob_no, 64, kernel_initializer=tf.truncated_normal_initializer(stddev=1.0)))
        net = tf.nn.tanh(tf.layers.dense(net, 64, kernel_initializer=tf.truncated_normal_initializer(stddev=1.0)))

        baseline_prediction= tf.layers.dense(net, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=1.0))

        return baseline_prediction

    def predict(self, ob):
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})

    def baseline_predict(self, ob):
        return self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no :ob[None]})

    def fit(self, ob_no, ac_na, adv_n, baseline_target_n):
        
        self.sess.run(self.update_op,
            feed_dict={self.sy_ob_no :ob_no, 
                       self.sy_ac_na:ac_na, 
                       self.sy_adv_n:adv_n,
                       self.sy_baseline_target_n: baseline_target_n})

    def assign_old_eq_new(self):
        return  self.sess.run(self.sync_old_network_params)