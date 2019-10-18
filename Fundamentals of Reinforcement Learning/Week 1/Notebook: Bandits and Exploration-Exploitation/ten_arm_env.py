#!/usr/bin/env python

from environment import BaseEnvironment

import numpy as np

class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [0]

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.arms = []
        self.seed = None

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        
        self.seed = env_info.get("random_seed", None)
        np.random.seed(self.seed)
        self.arms = np.random.randn(10)#[np.random.normal(0.0, 1.0) for _ in range(10)]
        local_observation = 0  # An empty NumPy array

        self.reward_obs_term = (0.0, local_observation, False)


    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        # if action == 0:
        #     if np.random.random() < 0.2:
        #         reward = 14
        #     else:
        #         reward = 6

        # if action == 1:
        #     reward = np.random.choice(range(10,14))
        
        # if action == 2:
        #     if np.random.random() < 0.8:
        #         reward = 174
        #     else:
        #         reward = 7

        # reward = np.random.normal(self.arms[action], 1.0)
        
        reward = self.arms[action] + np.random.randn()

        obs = self.reward_obs_term[1]

        self.reward_obs_term = (reward, obs, False)

        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
