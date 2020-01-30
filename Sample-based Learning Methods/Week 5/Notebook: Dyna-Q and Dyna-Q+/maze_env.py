#!/usr/bin/env python

from environment import BaseEnvironment

import numpy as np

class MazeEnvironment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.maze_dim = [6, 9]
        self.obstacles = [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]]

        self.start_state = [2, 0]
        self.end_state = [0, 8]
        self.current_state = [None, None]

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]

    def env_init(self, agent_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.get_observation(self.current_state)

        return self.reward_obs_term[1]

    # check if current state is within the gridworld and return bool
    def out_of_bounds(self, row, col):
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    # check if there is an obstacle at (row, col)
    def is_obstacle(self, row, col):
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # update current_state with the action (also check validity of action)
        if action == 0: # up
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        elif action == 1: # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2: # down
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3: # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

        if self.current_state == self.end_state: # terminate if goal is reached
            reward = 1.0
            is_terminal = True

        self.reward_obs_term = [reward, self.get_observation(self.current_state), is_terminal]

        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None

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




class ShortcutMazeEnvironment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.maze_dim = [6,9]
        self.obstacles = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8]]

        self.start_state = [5,3]
        self.end_state = [0,8]
        self.current_state = [None, None]

        # a shortcut opens up after n timesteps
        self.change_at_n = 0
        self.timesteps = 0

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.change_at_n = env_info.get('change_at_n', 100000)
        self.timesteps = 0
        self.reward_obs_term = [0.0, None, False]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.get_observation(self.current_state)

        return self.reward_obs_term[1]

    # check if current state is within the gridworld and return bool
    def out_of_bounds(self, row, col):
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    # check if there is an obstacle at (row, col)
    def is_obstacle(self, row, col):
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        self.timesteps += 1
        if self.timesteps == self.change_at_n:
            self.obstacles = self.obstacles[:-1]

        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # update current_state with the action (also check validity of action)
        if action == 0: # up
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        elif action == 1: # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2: # down
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3: # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

        if self.current_state == self.end_state: # terminate if goal is reached
            reward = 1.0
            is_terminal = True

        self.reward_obs_term = [reward, self.get_observation(self.current_state), is_terminal]

        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None

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
