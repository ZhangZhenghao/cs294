#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import load_policy
import tf_util


class NeuralNetwork:

    def __init__(self, obs_dim: int, hidden_units: int, action_dim: int):
        """
        Create a neural network with a single hidden layer.
        :param obs_dim: the dimension of observations
        :param hidden_units: the number of hidden units
        :param action_dim: the dimension of actions
        """
        # Placeholder
        self.sy_obs = tf.placeholder(tf.float32, shape=[None, obs_dim])
        self.sy_act = tf.placeholder(tf.float32, shape=[None, action_dim])
        # Network
        fc1 = tf.layers.dense(self.sy_obs, hidden_units, tf.nn.relu)
        self.sy_out = tf.layers.dense(fc1, action_dim)
        # Train step
        self.sy_loss = tf.losses.mean_squared_error(self.sy_act, self.sy_out)
        train_optimizer = tf.train.AdamOptimizer()
        self.train_step = train_optimizer.minimize(self.sy_loss)
        # Initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, observations, actions, iter, batch_size=1000, print_iter=100):
        """
        Train the neural network.
        :param observations: observations for training
        :param actions: actions for training
        :param iter: the number of training iterations
        :param batch_size: the size of training batch
        :param print_iter: when to print loss
        :return: the history of loss
        """
        train_size, _ = actions.shape
        loss_hist = []
        for i in range(iter):
            batch_index = np.random.choice(np.arange(0, train_size), batch_size)
            batch_obs = observations[batch_index]
            batch_acts = actions[batch_index]
            batch_loss, _ = self.sess.run([self.sy_loss, self.train_step], {
                self.sy_obs: batch_obs,
                self.sy_act: batch_acts
            })
            loss_hist.append(batch_loss)
            if i % print_iter == 0:
                print('iter %d/%d, loss %f' % (i, iter, batch_loss))
        return loss_hist

    def predict(self, observations):
        """
        Predict actions of observations.
        :param observations: observations for predicting
        :return: actions
        """
        return self.sess.run(self.sy_out, {
            self.sy_obs: observations
        })


class RecurrentNetwork:

    def __init__(self, time_step: int, obs_dim: int, hidden_units: int, action_dim: int):
        # Placeholder
        self.sy_obs = tf.placeholder(tf.float32, [None, time_step, obs_dim])
        self.sy_act = tf.placeholder(tf.float32, [None, action_dim])
        sy_obs_list = tf.unstack(self.sy_obs, axis=1)
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
        outputs, states = tf.nn.static_rnn(lstm, sy_obs_list, dtype=tf.float32)
        self.sy_out = tf.layers.dense(outputs[-1], action_dim)
        # Train step
        self.sy_loss = tf.losses.mean_squared_error(self.sy_act, self.sy_out)
        train_optimizer = tf.train.AdamOptimizer()
        self.train_step = train_optimizer.minimize(self.sy_loss)
        # Initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, observations, actions, iter, batch_size=1000, print_iter=100):
        train_size, _ = actions.shape
        loss_hist = []
        for i in range(iter):
            batch_index = np.random.choice(np.arange(0, train_size), batch_size)
            batch_obs = observations[batch_index]
            batch_acts = actions[batch_index]
            batch_loss, _ = self.sess.run([self.sy_loss, self.train_step], {
                self.sy_obs: batch_obs,
                self.sy_act: batch_acts
            })
            loss_hist.append(batch_loss)
            if i % print_iter == 0:
                print('iter %d/%d, loss %f' % (i, iter, batch_loss))
        return loss_hist

    def predict(self, observations):
        """
        Predict actions of observations.
        :param observations: observations for predicting
        :return: actions
        """
        return self.sess.run(self.sy_out, {
            self.sy_obs: observations
        })


def stdout_write(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def look_ahead(observations, num_step):
    num_batch, num_dim = observations.shape
    tempo_obs = np.zeros([num_batch, num_step, num_dim])
    for t in range(num_step):
        tempo_obs[:, -t-1, :] = np.roll(observations, t, 0)
        tempo_obs[:t, -t-1, :] = observations[0, :]
    return tempo_obs


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    parser.add_argument('--plot_loss', action='store_true')
    parser.add_argument('--dagger', type=int, default=0, help='Epoch of DAgger training')
    parser.add_argument('--tempo', type=int, default=0)
    args = parser.parse_args()
    # Load expert policy
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        stdout_write('Expert data generating ')

        returns_expert = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            stdout_write('.')
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                total_reward += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            returns_expert.append(total_reward)
        print()

        print('returns', returns_expert)
        print('mean return', np.mean(returns_expert))
        print('std of return', np.std(returns_expert))

        observations = np.array(observations)
        actions = np.squeeze(actions)

        obs_dim = observations.shape[1]
        act_dim = actions.shape[1]

        if args.tempo > 1:
            net = RecurrentNetwork(args.tempo, obs_dim, obs_dim, act_dim)
        else:
            net = NeuralNetwork(obs_dim, obs_dim, act_dim)
        returns_policy = []
        loss_hist = []
        for epoch in range(args.dagger+1):
            # Fit expert data
            if args.tempo > 1:
                loss_hist += net.fit(look_ahead(observations, args.tempo), actions, 10000, print_iter=1000)
            else:
                loss_hist += net.fit(observations, actions, 10000, print_iter=1000)
            # Print epoch
            stdout_write('Policy testing ' if epoch == args.dagger
                         else 'DAgger data generating (epoch %d/%d)' % (epoch+1, args.dagger))
            # Generate expert data
            for i in range(args.num_rollouts):
                stdout_write('.')
                obs = env.reset()
                done = False
                total_reward = 0.
                steps = 0
                tempo_obs = [obs] * args.tempo
                while not done:
                    # Generate actions by expert
                    if epoch < args.dagger:
                        action_expert = policy_fn(obs[None, :])
                        actions = np.append(actions, action_expert, 0)
                        observations = np.append(observations, [obs], 0)
                    # Generate observations by policy
                    if args.tempo > 1:
                        tempo_obs.append(obs)
                        action_policy = net.predict([tempo_obs[-args.tempo:]])
                    else:
                        action_policy = net.predict([obs])
                    obs, r, done, _ = env.step(action_policy)
                    total_reward += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps >= max_steps:
                        break
                if epoch == args.dagger:
                    returns_policy.append(total_reward)
            # Print new line
            print()
    # Print performance
    print('returns', returns_policy)
    print('mean return', np.mean(returns_policy))
    print('std of return', np.std(returns_policy))
    # Plot loss history
    if args.plot_loss:
        plt.plot(loss_hist)
        plt.show()


if __name__ == '__main__':
    main()
