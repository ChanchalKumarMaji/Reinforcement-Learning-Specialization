#!/usr/bin/env python

from agent import BaseAgent

import numpy as np

class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.epsilon = None
        self.initial_value = 0.0
        self.arm_count = [0.0 for _ in range(10)]

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        # if "actions" in agent_info:
        #     self.num_actions = agent_info["actions"]

        # if "state_array" in agent_info:
        #     self.q_values = agent_info["state_array"]

        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(agent_info.get("num_actions", 2)) * self.initial_value
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)

        self.last_action = 0

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.last_action = np.random.choice(self.num_actions)  # set first action to 0

        return self.last_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # local_action = 0  # choose the action here
        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass