import gymnasium as gym
import numpy as np
import torch
from discriminator import Discriminator
from collections import deque

class GAILWrapper(gym.Wrapper):
    """
    A simple reward wrapper that demonstrates how to modify rewards.
    This wrapper adds a small bonus for each step and applies a penalty for large actions.
    """
    
    def __init__(self, env, discriminator: Discriminator):
        super().__init__(env)
        self.disc = discriminator
        self.current_state = None
        self.frame_buffer = deque(maxlen=4)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_state = obs  # Store the initial state
        for _ in range(4):
            self.frame_buffer.append(obs)
        return obs, info
    
    def get_frame_buffer(self):
        """
        Get the current frame buffer as a numpy array.
        Returns: numpy array of shape (4, H, W, C) containing the last 4 frames
        """
        return np.array(list(self.frame_buffer))
    
    def get_frame_buffer_tensor(self):
        """
        Get the current frame buffer as a torch tensor formatted for the discriminator.
        Returns: torch tensor of shape (1, 4, 3, H, W) ready for discriminator input
        """
        frame_array = np.array(list(self.frame_buffer))  # (4, H, W, C)
        frame_tensor = torch.tensor(frame_array, dtype=torch.float32)
        
        # Rearrange from (4, H, W, C) to (1, 4, 3, H, W) for discriminator
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 4, 3, H, W)
        
        return frame_tensor

    def step(self, action):
        # Store the previous 4 frames before taking the action
        previous_state_buffer = list(self.frame_buffer)
        
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_state = obs  # Update current state
        self.frame_buffer.append(obs)
        
        # Convert to tensor: from list of (H, W, C) to (4, H, W, C)
        previous_state_buffer = torch.tensor(np.array(previous_state_buffer), dtype=torch.float32)
        
        # Add batch dimension and ensure correct format for discriminator
        # Expected format: (B, 4, 3, 96, 96) where B=1, 4 frames, 3 RGB channels
        if len(previous_state_buffer.shape) == 4:  # (4, H, W, C)
            # Rearrange from (4, H, W, C) to (1, 4, 3, H, W) 
            previous_state_buffer = previous_state_buffer.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 4, 3, H, W)
          
        
        action = torch.tensor(action, dtype=torch.float32)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)  # Add batch dimension

        reward = self.disc.reward(previous_state_buffer, action)
        return obs, reward, terminated, truncated, info







