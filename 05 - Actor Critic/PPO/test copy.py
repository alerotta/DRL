import gymnasium as gym
import torch
from Network import PPONetwork
from collections import deque 

if __name__ == "__main__" :

    env = gym.make("CarRacing-v3")
    print(env.observation_space.shape[-1] , env.action_space.shape[0])

    net = PPONetwork(env.observation_space.shape,  env.action_space.shape[0])
    
    state , _ = env.reset()
    d = deque(maxlen=2)
    tensor = torch.tensor(state , dtype=torch.float32).permute(2,1,0).unsqueeze(0)
    tensor1 = torch.tensor(state , dtype=torch.float32).permute(2,1,0).unsqueeze(0)
    d.append(tensor)
    d.append(tensor1)
    print(list(d))



    


