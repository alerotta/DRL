import gymnasium as gym
import numpy as np
import torch
from discriminator import Discriminator

class GAILWrapper(gym.Wrapper):
    """
    A simple reward wrapper that demonstrates how to modify rewards.
    This wrapper adds a small bonus for each step and applies a penalty for large actions.
    """
    
    def __init__(self, env, discriminator: Discriminator):
        super().__init__(env)
        self.disc = discriminator
        self.current_state = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_state = obs  # Store the initial state
        return obs, info
    
    def step(self, action):

        previous_state = self.current_state
        self.current_state, _ , terminated , trunacted ,  info  = self.env.step(action)
        
        previous_state = torch.tensor(previous_state, dtype=torch.float32)
        # If state is an image with shape [H, W, C], convert to [C, H, W] for PyTorch
        if len(previous_state.shape) == 3 and previous_state.shape[-1] == 3:
            previous_state = previous_state.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        # Add batch dimension if needed
        if len(previous_state.shape) == 3:
            previous_state = previous_state.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        action = torch.tensor(action, dtype=torch.float32)
        action = action.unsqueeze(0)

        reward = self.disc.reward(previous_state ,action)
        return  self.current_state, reward , terminated , trunacted ,  info
    

    




