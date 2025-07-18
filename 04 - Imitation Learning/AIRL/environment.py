import gymnasium as gym
import torch
import numpy as np
from collections import deque
from discriminator import AIRLDiscriminator

class EnvWrapper (gym.Wrapper):

    def __init__(self, env, discriminator: AIRLDiscriminator):
        super().__init__(env)
        self.disc = discriminator
        self.frame_buffer = deque(maxlen=4)

        self.obsv , self.info  = self.env.reset()
        for _ in range(4):
            self.frame_buffer.append(self.obsv)
        self.state =  np.array([np.stack(self.frame_buffer)], dtype=np.float32)
        self.state_tensor = torch.FloatTensor(self.state)
        self.state_tensor = self.state_tensor.view(self.state_tensor.size(0) ,-1 , 96 ,96)
        self.state_tensor = self.state_tensor / 255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(4):
            self.frame_buffer.append(obs)
        self.state =  np.array([np.stack(self.frame_buffer)], dtype=np.float32)
        self.state_tensor = torch.FloatTensor(self.state)
        self.state_tensor = self.state_tensor.view(self.state_tensor.size(0) ,-1 , 96 ,96)
        self.state_tensor = self.state_tensor / 255.0
        return obs, info

        

    def step(self,action):

        self.state =  np.array([np.stack(self.frame_buffer)], dtype=np.float32)
        self.state_tensor = torch.FloatTensor(self.state)
        self.state_tensor = self.state_tensor.view(self.state_tensor.size(0) ,-1 , 96 ,96)
        self.state_tensor = self.state_tensor / 255.0
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        reward = self.disc.get_logits(self.state_tensor,action_tensor)
        obsv , _ , is_terminated , is_truncated , info = self.env.step(action)
        self.frame_buffer.append(obsv)

        return obsv , reward, is_terminated , is_truncated , info
    
    def get_state_tensor (self) :
        return self.state_tensor