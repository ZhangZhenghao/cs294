import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

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
        assert len(env.action_space.shape) == 1
        assert len(env.observation_space.shape) == 1
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess
        # Fetch normalization parameters
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = normalization
        # Get action space and obs space
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # Build MLP
        self.obs_input = tf.placeholder(tf.float32, [None, obs_dim])
        self.act_input = tf.placeholder(tf.float32, [None, act_dim])
        self.delta_input = tf.placeholder(tf.float32, [None, obs_dim])
        obs_norm_input = (self.obs_input - mean_obs) / (std_obs + 1e-5)
        act_norm_input = (self.act_input - mean_action) / (std_action + 1e-5)
        delta_norm_input = (self.delta_input - mean_deltas) / (std_deltas + 1e-5)
        concat_norm_input = tf.concat([obs_norm_input, act_norm_input], 1)
        delta_norm_output = build_mlp(concat_norm_input, obs_dim, "dynamic", n_layers, size, activation, output_activation)
        # Build train step
        loss = tf.losses.mean_squared_error(delta_norm_input, delta_norm_output)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # Compute next state
        delta_output = mean_deltas + delta_norm_output * std_deltas
        self.next_obs_output = self.obs_input + delta_output

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        obs = data['observations']
        deltas = data['deltas']
        action = data['actions']
        num_train = len(obs)
        for i in range(self.iterations):
            batch_index = np.random.choice(num_train, self.batch_size)
            batch_obs = obs[batch_index]
            batch_deltas = deltas[batch_index]
            batch_action = action[batch_index]
            self.sess.run(self.train_step, {
                self.obs_input: batch_obs,
                self.delta_input: batch_deltas,
                self.act_input: batch_action
            })

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        return self.sess.run(self.next_obs_output, {
            self.obs_input: states,
            self.act_input: actions
        })
