#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np

class DummyEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        
        reward = 0.0
        observation = None
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

        reward = 0.0
        is_terminal = True
        current_state = None
        
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term