#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np

class RandomWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        
        Set parameters needed to setup the 500-state random walk environment.
        
        Assume env_info dict contains:
        {
            num_states: 500,
            start_state: 250,
            left_terminal_state: 0,
            right_terminal_state: 501,
            seed: int
        }
        """
        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed")) 
        
        ### Set each attributes correctly (4 lines)
        # self.num_states = ?
        # self.start_state = ?
        # self.left_terminal_state = ?
        # self.right_terminal_state = ?
        
        ### START CODE HERE ### 
        
        self.num_states = env_info["num_states"] 
        self.start_state = env_info["start_state"] 
        self.left_terminal_state = env_info["left_terminal_state"] 
        self.right_terminal_state = env_info["right_terminal_state"]
        
        ### END CODE HERE ###
        
        

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        ### set self.reward_obs_term tuple accordingly (3 lines)
        # reward = ?
        # observation = ?
        # is_terminal = ?
        
        ### START CODE HERE ### 3 lines
        
        
        reward = 0.0
        observation = self.start_state
        is_terminal = False
        
        ### END CODE HERE ###
        
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
        
        ### set reward, current_state, and is_terminal correctly (10~12 lines)
        # current state: next state after taking action from the last state [int]
        # action: represents how many states to move from the last state [int]
        # Hint: Given action (direction of movement), determine how much to move in that direction 
        #       by calling self.rand_generator.choice() once. 
        #       Solutions using other random methods may be graded as incorrect.
        #       Remember all transitions beyond the terminal state is absorbed into the terminal state.
        #
        # reward = ?
        # current_state = ?
        # is_terminal = ?
        
        ### START CODE HERE ### 

        last_state = self.reward_obs_term[1]
        
        if action == 0: # left
            current_state = max(self.left_terminal_state, last_state + self.rand_generator.choice(range(-100,0)))
        elif action == 1: # right
            current_state = min(self.right_terminal_state, last_state + self.rand_generator.choice(range(1,101)))
        else: 
            raise ValueError("Wrong action value")
        
        reward = 0.0
        is_terminal = False
        if current_state == self.left_terminal_state: 
            reward = -1.0
            is_terminal = True

        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True

        ### END CODE HERE ###
        
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term