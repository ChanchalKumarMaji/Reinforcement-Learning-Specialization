#!/usr/bin/env python
import numpy as np
from agent import BaseAgent

class DummyAgent(BaseAgent):
    def __init__(self):
        self.name = "dummy_agent"
        self.step_size = None
        self.discount_factor = None
        pass
       
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) with Neural Network.

        Assume agent_info dict contains: TODO
        {
            step_size: float, 
            discount_factor: float, 
        }
        """

        self.rand_generator = np.random.RandomState(agent_info.get("seed")) # set random seed for each run

        # save relevant info from agent_info
        self.input_dim = agent_info.get("input_dim")
        self.num_actions = agent_info.get("num_actions")
        self.step_size = agent_info.get("step_size")
        self.discount_factor = agent_info.get("discount_factor")
        self.tau = agent_info.get("tau")
        
        self.weights = np.zeros((self.num_actions, self.input_dim))

        self.last_state = None
        self.last_action = None


    def choose_action(self, observation):
        return self.rand_generator.randint(self.num_actions)

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        
        self.last_state = observation
        self.last_action = self.choose_action(observation)

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

        state = observation

        # choose action
        action = self.choose_action(state)

        self.last_state = observation
        self.last_action = action
        
        return action


    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        return

        
    def agent_message(self, message):
        if message == 'get_sum_reward':
            return self.rand_generator.normal(0, 0.1) * -1 * (np.log2(self.step_size / 3e-5) ** 2 + np.log(self.tau) ** 2)
