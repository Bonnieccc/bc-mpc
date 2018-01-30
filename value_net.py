import numpy as np
import tensorflow as tf
from utils import build_mlp


class value_network(object):
    """docstring for policy_network"""
    def __init__(self, sess, ob_dim,  n_layers, size, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size 
        
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        self.baseline_prediction = self.build_network(self.sy_ob_no)
        self.sy_baseline_target_n = tf.placeholder(shape=[None], name="baseline_target", dtype=tf.float32)

        # setup loss

        self.loss_baseline = tf.reduce_mean(tf.square(self.baseline_prediction - self.sy_baseline_target_n))
        self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_baseline)


    def build_network(self, sy_ob_no):
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1, 
                                "value_network",
                                n_layers=self.n_layers,
                                size=self.size))
        return baseline_prediction

    def predict(self, ob_no):
        return self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no :ob_no})

    def fit(self, ob_no, b_n_target):
        self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no :ob_no, self.sy_baseline_target_n:b_n_target})


