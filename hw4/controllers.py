import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        super().__init__()
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        super().__init__()
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        action_dim = self.env.action_space.shape[0]
        # return np.zeros(self.env.action_space.shape)
        state_dim = self.env.observation_space.shape[0]
        path_actions = np.zeros([self.horizon, self.num_simulated_paths, action_dim])
        path_states = np.zeros([self.horizon, self.num_simulated_paths, state_dim])
        path_next_states = np.zeros([self.horizon, self.num_simulated_paths, state_dim])
        states = np.ones([self.num_simulated_paths, state_dim]) * state.reshape([-1, state_dim])
        for i in range(self.horizon):
            path_states[i] = state
            path_actions[i] = np.asarray([self.env.action_space.sample() for _ in range(self.num_simulated_paths)])
            states = self.dyn_model.predict(states, path_actions[i])
            path_next_states[i] = states
        path_costs = trajectory_cost_fn(self.cost_fn, path_states, path_actions, path_next_states)
        best = np.argmin(path_costs)
        return path_actions[0, best]


