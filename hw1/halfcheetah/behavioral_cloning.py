#!/usr/bin/env python
import tensorflow as tf
import tflearn
import numpy as np
import gym
import load_policy

# ===========================
# Training parameters
# ===========================
LEARNING_RATE = 1e-3
TRAIN_EPOCH = 50000
TEST_EPOCH = 100
BATCH_SIZE = 128
DAGGER = True
# ==========================
# Environment
# ==========================
envname = 'HalfCheetah-v1'
expert_policy_file = 'experts/HalfCheetah-v1.pkl' 
num_rollouts = 10
render = False
# ===========================
# Build network
# ===========================
class BCnetwork(object):

    def __init__(self, sess, obs_dim, act_dim, learning_rate):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = learning_rate
        self.exp_actions = tf.placeholder(tf.float32, [None, self.act_dim])
        self.obs, self.actions = self.creat_bc_network()
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.actions, self.exp_actions))))
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=1e-6, centered=True).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    def creat_bc_network(self):        
        observations = tf.placeholder(tf.float32, shape = [None, self.obs_dim])


        net = tflearn.fully_connected(observations, 128, activation='tanh')
        net = tflearn.fully_connected(net, 64, activation='tanh')
        actions = tflearn.fully_connected(net, self.act_dim)

        # net = tf.layers.dense(observations, 128, activation=tf.nn.tanh)
        # net = tf.layers.dense(net, 64, activation=tf.nn.tanh)
        # actions = tf.layers.dense(net, self.act_dim)

        return observations, actions

    def train(self, exp_obs, exp_actions):
        return self.sess.run([self.actions, self.optimizer, self.loss], feed_dict={
            self.obs: exp_obs,
            self.exp_actions: exp_actions,
        })

    def predict(self, obs):
        return self.sess.run(self.actions, feed_dict={
            self.obs: obs
        })

def train(sess, bc_net, expert_data, env, max_steps):
    # load expert 
    policy_fn = load_policy.load_policy(expert_policy_file)
    print("now training data size: ", expert_data['observations'].shape[0])

    sess.run(tf.global_variables_initializer())
    EP = 0
    if DAGGER:
        print("train in Dagger mode")

    while EP < TRAIN_EPOCH:
        EP += 1
        sample_size = expert_data['observations'].shape[0]
        batch_idx = np.random.randint(sample_size, size=BATCH_SIZE)
        batch_obs = expert_data['observations'][batch_idx, :]
        batch_actions = expert_data['actions'][batch_idx, :]
        
        if DAGGER and EP%500 ==0:
            print("Daggering")
            observations = []
            actions = []
            for i in range(num_rollouts):
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    action = bc_net.predict(obs[None,:])
                    observations.append(obs)
                    expert_action = policy_fn(obs[None,:])
                    actions.append(expert_action)
                    obs, r, done, _ = env.step(action)
                    steps += 1
                    if steps >= max_steps:
                        break
            # aggregate data
            new_obs = np.array(observations)
            new_act = np.array(actions)
            new_act = new_act.reshape(new_act.shape[0], -1)
            expert_data['observations'] = np.concatenate((expert_data['observations'], new_obs), axis=0)
            expert_data['actions'] = np.concatenate((expert_data['actions'], new_act), axis=0)
            print("now training data size: ", expert_data['observations'].shape[0])

        _, _, loss = bc_net.train(batch_obs, batch_actions)

        if EP % 100 == 0:
            print('epcho: ', EP, ' loss: ', loss)

        if EP % TEST_EPOCH == 0:
            test(sess, bc_net, env, max_steps)

def test(sees, bc_net, env, max_steps):

    returns = []
    observations = []
    actions = []

    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = bc_net.predict(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        # print('iter', i, ' totalr', totalr)

        returns.append(totalr)

    # print('returns', returns)
    print('----------- bc performance ---------- ')
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


def main():

# ===========================
# load expert_data
# ===========================
    expert_data = {}
    expert_data['actions'] = np.load("./expert_actions.npy")
    expert_data['observations'] = np.load("./expert_observations.npy")
    expert_data['actions'] = expert_data['actions'].reshape(expert_data['actions'].shape[0], -1)
    print("successfully load expert_data")
    print("observation dim: ", expert_data['observations'].shape)
    print("action dim: ", expert_data['actions'].shape)
    obs_dim = expert_data['observations'].shape[1]
    act_dim = expert_data['actions'].shape[1]

# ==============================
# Configure the training process
# ==============================
    env = gym.make(envname)
    max_steps = env.spec.timestep_limit

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        bc_net = BCnetwork(sess, obs_dim, act_dim, LEARNING_RATE)
        train(sess, bc_net, expert_data, env, max_steps)
        test(sess, bc_net, env, max_steps)





if __name__ == '__main__':
    main()
