import tensorflow as tf
import numpy as np

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.states_input_placeholder =  tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        self.actions_input_placeholder =  tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape[0]))

        self.states_action_input = tf.concat([self.states_input_placeholder, self.actions_input_placeholder], axis=1)
        # self.nxt_states_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        self.states_delta = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))

        # print("input_placeholder: ", self.input_placeholder)
        self.scope = "NNDynamicsModel"
        self.state_delta_predict = self.build_network(self.states_action_input, 
                                   self.env.observation_space.shape[0], 
                                   self.scope, 
                                   n_layers=n_layers, 
                                   size=size,
                                   activation=activation,
                                   output_activation=output_activation)

        # data normalization
        # self.mean_obs, self.std_obs, self.mean_action, self.std_action, self.mean_reward, self.std_reward, self.mean_nxt_state, self.std_nxt_state, self.mean_deltas, self.std_deltas = normalization
        self.mean_obs, self.std_obs, self.mean_action, self.std_action, self.mean_nxt_state, self.std_nxt_state, self.mean_deltas, self.std_deltas = normalization

        # optimization
        self.sess = sess
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size

        # states_delta = self.nxt_states_placeholder - self.states_input_placeholder
        self.loss = tf.reduce_mean(tf.squared_difference(self.states_delta, self.state_delta_predict))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_network(self,
                  input_placeholder, 
                  output_size,
                  scope, 
                  n_layers=2, 
                  size=500, 
                  activation=tf.tanh,
                  output_activation=None
                  ):
        # Predefined function to build a feedforward neural network
        out = input_placeholder
        with tf.variable_scope(scope):
            for _ in range(n_layers):
                out = tf.layers.dense(out, size, activation=activation)
            out = tf.layers.dense(out, output_size, activation=output_activation)
        return out

    def normalize(self, unnormalized_data, std, mean):
        normalized_data =  (unnormalized_data - mean)/ (std+ 1e-10)
        return normalized_data

    def denomalize(self, normalized_data, std, mean):
        unnormalized_data =  (normalized_data * std) + mean
        return unnormalized_data

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        print("Model fitting for ", self.iterations, "times ... ")
        for i in range(self.iterations):
            # print("dynamic fit iter: ", i)
            # sample_state, sample_action, sample_reward, sample_nxt_state, sample_state_delta = data.sample(self.batch_size)
            sample_state, sample_action, sample_nxt_state, sample_state_delta = data.sample(self.batch_size)

            normalized_sample_state =  self.normalize(sample_state, self.std_obs, self.mean_obs)
            normalized_sample_action = self.normalize(sample_action, self.std_action, self.mean_action)
            normalized_sample_nxt_state =  self.normalize(sample_nxt_state, self.std_obs, self.mean_obs)
            normalized_sample_state_delta =  self.normalize(sample_state_delta, self.std_deltas, self.mean_deltas)

            loss, _ = self.sess.run([self.loss, self.train_step], 
                          feed_dict={self.states_input_placeholder:normalized_sample_state, 
                                     self.actions_input_placeholder:normalized_sample_action,
                                     self.states_delta:normalized_sample_state_delta})
            
            # print("loss ", i, " : ", loss)

    def predict(self, unnormalized_state, unnormalized_action):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        normalized_state =  (unnormalized_state - self.mean_obs)/ (self.std_obs + 1e-10)
        normalized_action =  (unnormalized_action - self.mean_action)/ (self.std_action + 1e-10)

        normalized_state_delta = self.sess.run(self.state_delta_predict, feed_dict={self.states_input_placeholder:normalized_state, self.actions_input_placeholder:normalized_action})
        unnormalized_state_delta = self.denomalize(normalized_state_delta, self.std_deltas, self.mean_deltas)


        unnormalized_nxt_state = unnormalized_state + unnormalized_state_delta


        return unnormalized_nxt_state

class NNDynamicsRewardModel():
    def __init__(self, 
                 env, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.states_input_placeholder =  tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        self.actions_input_placeholder =  tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape[0]))

        self.states_action_input = tf.concat([self.states_input_placeholder, self.actions_input_placeholder], axis=1)
        # self.nxt_states_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        self.states_delta = tf.placeholder(tf.float32, shape=(None, self.env.observation_space.shape[0]))
        self.reward = tf.placeholder(tf.float32, shape=(None, 1))

        # print("input_placeholder: ", self.input_placeholder)
        self.scope = "NNDynamicsModel"
        self.state_delta_predict, self.reward_predict = self.build_network(self.states_action_input, 
                                   self.env.observation_space.shape[0], 
                                   self.scope)
        # data normalization
        self.mean_obs, self.std_obs, self.mean_action, self.std_action, self.mean_reward, self.std_reward, self.mean_nxt_state, self.std_nxt_state, self.mean_deltas, self.std_deltas = normalization

        # optimization
        self.sess = sess
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size

        self.loss_dynamic = tf.reduce_mean(tf.squared_difference(self.states_delta, self.state_delta_predict))
        self.loss_reward = tf.reduce_mean(tf.squared_difference(self.reward, self.reward_predict))
        self.loss = self.loss_dynamic + self.loss_reward

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_network(self, states_action_input, state_size, scope, activation=tf.tanh):
        # Predefined function to build a feedforward neural network with dynamic and reward
        with tf.variable_scope(scope):
            # share layer
            share = tf.layers.dense(states_action_input, 500, activation=activation)

            # state delta prediction
            state_delta_predict = tf.layers.dense(share, 500, activation=activation)
            state_delta_predict = tf.layers.dense(state_delta_predict, state_size, activation=None)

            # reward prediction
            reward_predict = tf.layers.dense(share, 500, activation=activation)
            reward_predict = tf.layers.dense(reward_predict, 1, activation=None)

        return state_delta_predict, reward_predict

    def normalize(self, unnormalized_data, std, mean):
        normalized_data =  (unnormalized_data - mean)/ (std+ 1e-10)
        return normalized_data

    def denomalize(self, normalized_data, std, mean):
        unnormalized_data =  (normalized_data * std) + mean
        return unnormalized_data

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        print("Model fitting for ", self.iterations, "times ... ")
        for i in range(self.iterations):
            # print("dynamic fit iter: ", i)
            sample_state, sample_action, sample_reward, sample_nxt_state, sample_state_delta = data.sample(self.batch_size)

            normalized_sample_state =  self.normalize(sample_state, self.std_obs, self.mean_obs)
            normalized_sample_action = self.normalize(sample_action, self.std_action, self.mean_action)
            normalized_sample_reward =  self.normalize(sample_reward, self.std_reward, self.mean_reward)
            normalized_sample_nxt_state =  self.normalize(sample_nxt_state, self.std_obs, self.mean_obs)
            normalized_sample_state_delta =  self.normalize(sample_state_delta, self.std_deltas, self.mean_deltas)

            loss, _ = self.sess.run([self.loss, self.train_step], 
                          feed_dict={self.states_input_placeholder:normalized_sample_state, 
                                     self.actions_input_placeholder:normalized_sample_action,
                                     self.reward:np.reshape(normalized_sample_reward, [-1, 1]),
                                     self.states_delta:normalized_sample_state_delta})
            
            # print("loss ", i, " : ", loss)

    def predict(self, unnormalized_state, unnormalized_action):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        normalized_state =  (unnormalized_state - self.mean_obs)/ (self.std_obs + 1e-10)
        normalized_action =  (unnormalized_action - self.mean_action)/ (self.std_action + 1e-10)


        normalized_state_delta, normalized_reward = self.sess.run([self.state_delta_predict, self.reward_predict],
                                 feed_dict={self.states_input_placeholder:normalized_state, 
                                            self.actions_input_placeholder:normalized_action,
                                            })

        unnormalized_state_delta = self.denomalize(normalized_state_delta, self.std_deltas, self.mean_deltas)
        unnormalized_nxt_state = unnormalized_state + unnormalized_state_delta

        unnormalized_reward = self.denomalize(normalized_reward, self.std_reward, self.mean_reward)

        return unnormalized_nxt_state, unnormalized_reward






