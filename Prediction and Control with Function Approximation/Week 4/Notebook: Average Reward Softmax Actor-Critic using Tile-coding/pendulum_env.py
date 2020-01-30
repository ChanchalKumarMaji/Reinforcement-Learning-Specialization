from environment import BaseEnvironment
import numpy as np

class PendulumEnvironment(BaseEnvironment):
    
    def __init__(self):
        self.rand_generator = None
        self.ang_velocity_range = None
        self.dt = None
        self.viewer = None
        self.gravity = None
        self.mass = None
        self.length = None
        
        self.valid_actions = None
        self.actions = None
        
    
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        
        Set parameters needed to setup the pendulum swing-up environment.
        """
        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed"))     
        
        self.ang_velocity_range = [-2 * np.pi, 2 * np.pi]
        self.dt = 0.05
        self.viewer = None
        self.gravity = 9.8
        self.mass = float(1./3.)
        self.length = float(3./2.)
        
        self.valid_actions = (0,1,2)
        self.actions = [-1,0,1]
        
        self.last_action = None
    
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        ### set self.reward_obs_term tuple accordingly (3~5 lines)
        # Angle starts at -pi or pi, and Angular velocity at 0.
        # reward = ?
        # observation = ?
        # is_terminal = ?
        
        beta = -np.pi
        betadot = 0.
        
        reward = 0.0
        observation = np.array([beta, betadot])
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
        
        ### set reward, observation, and is_terminal correctly (10~12 lines)
        # Update the state according to the transition dynamics
        # Remember to normalize the angle so that it is always between -pi and pi.
        # If the angular velocity exceeds the bound, reset the state to the resting position
        # Compute reward according to the new state, and is_terminal should always be False
        # 
        # reward = ?
        # observation = ?
        # is_terminal = ?

        # Check if action is valid
        assert(action in self.valid_actions)
        
        last_state = self.reward_obs_term[1]
        last_beta, last_betadot = last_state        
        self.last_action = action
        
        betadot = last_betadot + 0.75 * (self.actions[action] + self.mass * self.length * self.gravity * np.sin(last_beta)) / (self.mass * self.length**2) * self.dt

        beta = last_beta + betadot * self.dt

        # normalize angle
        beta = ((beta + np.pi) % (2*np.pi)) - np.pi
        
        # reset if out of bound
        if betadot < self.ang_velocity_range[0] or betadot > self.ang_velocity_range[1]:
            beta = -np.pi
            betadot = 0.
        
        # compute reward
        reward = -(np.abs(((beta+np.pi) % (2 * np.pi)) - np.pi))
        observation = np.array([beta, betadot])
        is_terminal = False

        
        self.reward_obs_term = (reward, observation, is_terminal)
        
        return self.reward_obs_term