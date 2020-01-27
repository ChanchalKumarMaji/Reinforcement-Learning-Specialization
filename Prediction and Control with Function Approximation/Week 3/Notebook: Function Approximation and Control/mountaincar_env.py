#!/usr/bin/env python

from environment import BaseEnvironment

import numpy as np

class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment
    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [0, 1, 2]

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.current_state = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0

    def env_init(self, agent_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        local_observation = 0  # An empty NumPy array

        self.reward_obs_term = (0.0, local_observation, False)


    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """

        position = np.random.uniform(-0.6, -0.4)
        velocity = 0.0
        self.current_state = np.array([position, velocity]) # position, velocity

        return self.current_state

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        position, velocity = self.current_state

        terminal = False
        reward = -1.0
        velocity = self.bound_velocity(velocity + 0.001 * (action - 1) - 0.0025 * np.cos(3 * position))
        position = self.bound_position(position + velocity)

        if position == -1.2:
            velocity = 0.0
        elif position == 0.5:
            self.current_state = None
            terminal = True
            reward = 0.0

        self.current_state = np.array([position, velocity])

        self.reward_obs_term = (reward, self.current_state, terminal)

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

    def bound_velocity(self, velocity):
        if velocity > 0.07:
            return 0.07
        if velocity < -0.07:
            return -0.07
        return velocity

    def bound_position(self, position):
        if position > 0.5:
            return 0.5
        if position < -1.2:
            return -1.2
        return position