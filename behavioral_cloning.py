import tensorflow as tf
import tflearn
import numpy as np

# ===========================
# Build network
# ===========================
class BCnetwork(object):

    def __init__(self, sess, env, batch_size, learning_rate):
        self.sess = sess
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exp_actions = tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape[0]))
        self.obs, self.actions = self.creat_bc_network()
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.actions, self.exp_actions))))
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=1e-6, centered=True).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def creat_bc_network(self):        
        observations = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        net = tflearn.fully_connected(observations, 128, activation='tanh')
        net = tflearn.fully_connected(net, 64, activation='tanh')
        actions = tflearn.fully_connected(net, self.env.action_space.shape[0])
        return observations, actions

    def train(self, data, steps=1):

        losses = []
        for i in range(steps):
            sample_state, sample_action = data.sample(self.batch_size)
            # print("sample_state: ", sample_state.shape)
            # print("sample_action: ", sample_action.shape)
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={
                self.obs: sample_state,
                self.exp_actions: sample_action,
            })
            losses.append(loss)
            # print("loss: ", loss)

        return np.mean(losses)

    def predict(self, obs):
        return self.sess.run(self.actions, feed_dict={
            self.obs: obs
        })