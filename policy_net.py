import numpy as np
import tensorflow as tf
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

        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_sampled_ac, self.sy_logprob_n = self.build_network()

        # setup loss
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build_network(self):
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

        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_sampled_ac, self.sy_logprob_n = self.build_network()

        # setup loss
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # for trpo
        self.oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
        self.oldlogstd_a  = tf.placeholder(name="oldlogstd", shape=[ac_dim], dtype=tf.float32)
        self.oldlogstd_na = tf.ones(shape=(self.n,ac_dim), dtype=tf.float32) * self.oldlogstd_a

        # KL divergence and entropy among Gaussian(s).
        self.kl  = tf.reduce_mean(utils.gauss_KL(self.sy_mean_na, self.sy_logstd, self.oldmean_na, self.oldlogstd_na))
        self.ent = 0.5 * self.ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(self.sy_logstd)

    def build_network(self):
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 

        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        self.sy_logstd_a = tf.Variable(tf.zeros([self.ac_dim]), name='action/logstd', dtype=tf.float32) # logstd should just be a trainable variable, not a network output.
        self.sy_logstd_na = tf.Variable(tf.zeros([1, self.ac_dim]), name='action/logstd', dtype=tf.float32) # logstd should just be a trainable variable, not a network output.

        net = sy_ob_no

        net = tf.layers.dense(net, 64, activation=tf.tanh)

        net = tf.layers.dense(net, 64, activation=tf.tanh)


        self.sy_mean_na = tf.layers.dense(net, self.ac_dim, activation=None)

        self.sy_std = tf.exp(self.sy_logstd)
        # print('self.sy_std', self.sy_std)

        # Sample an action
        sy_sampled_ac = self.sy_mean_na + self.sy_std * tf.random_normal(tf.shape(self.sy_mean_na))
        # print('sy_sampled_ac', sy_sampled_ac)

        # Log likely hood of chosen this action
        # Hint: Use the log probability under a multivariate gaussian.
        sy_z = (sy_ac_na - self.sy_mean_na)/self.sy_std
        # print('sy_z', sy_z)
        sy_logprob_n = -0.5 * tf.square(sy_z) - 0.5 * tf.log(tf.constant(2*np.pi)) - self.sy_logstd

        sy_logprob_n = tf.reduce_sum(sy_logprob_n, axis=1)
        # print('sy_logprob_n', sy_logprob_n)

        return sy_ob_no, sy_ac_na, sy_adv_n, sy_sampled_ac, sy_logprob_n

    def predict(self, ob):
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})

    def fit(self, ob_no, ac_na, adv_n):
        loss, oldmean_na, oldlogstd_a, _ =  self.sess.run(
            [self.loss, self.sy_mean_na, self.sy_logstd_na, self.update_op],
            feed_dict={self.sy_ob_no :ob_no, 
                       self.sy_ac_na:ac_na, 
                       self.sy_adv_n:adv_n})

    def kldiv_and_entropy(self, ob_no, oldmean_na, oldlogstd_a):
        """ Returning KL diverence and current entropy since they can re-use
        some of the computation. For the KL divergence, though, we reuqire the
        old mean *and* the old log standard deviation to fully characterize the
        set of probability distributions we had earlier, each conditioned on
        different states in the MDP. Then we take the *average* of these, etc.,
        similar to the discrete case.
        """
        feed = {self.ob_no: ob_no,
                self.oldmean_na: oldmean_na,
                self.oldlogstd_a: oldlogstd_a}
        return self.sess.run([self.kl, self.ent], feed_dict=feed)
