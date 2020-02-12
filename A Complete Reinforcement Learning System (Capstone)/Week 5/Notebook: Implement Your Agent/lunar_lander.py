#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np
import gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env = gym.make("LunarLander-v2")
        self.env.seed(0)

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        
        reward = 0.0
        observation = self.env.reset()
        is_terminal = False
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term[1]
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)
        
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term