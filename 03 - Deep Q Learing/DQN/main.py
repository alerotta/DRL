import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym 
import numpy as np

from dataclasses import dataclass
from collections import deque
from network import CnnNetwork

import ale_py

REPLAY_SIZE = 10000
GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


class ExpBuffer () : 
    def __init__(self, size): 
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)
    
    def append (self,x):
        self.buffer.append(x)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
class ExperienceDataset(Dataset):
    def __init__(self, experience_buffer):
        self.experiences = list(experience_buffer.buffer)
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        return {
            'state': exp.state,
            'action': exp.action,
            'reward': exp.reward,
            'is_done_trunc': exp.is_done_trunc,
            'next_state': exp.next_state
        }
    
@dataclass
class Experience:
    state: np.array
    action: int 
    reward: float 
    is_done_trunc: bool
    next_state: np.array



class Agent():

    def __init__(self, env : gym.Env):

        self.env = env 
        self.replay_buffer = ExpBuffer(REPLAY_SIZE)
        self.network = CnnNetwork(env.observation_space.shape,env.action_space.n)
        self.target_network = CnnNetwork(env.observation_space.shape,env.action_space.n)
        self.epsilon = EPSILON_START
        self.frame_count = 0  # Track total frames for epsilon decay

    @torch.no_grad()
    def play_episode (self):

        state , _ = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.permute(2, 0, 1)  
        total_reward = 0.0
        
        while True:
            # Decay epsilon based on frame count
            if self.frame_count < EPSILON_DECAY_LAST_FRAME:
                self.epsilon = EPSILON_START - (EPSILON_START - EPSILON_FINAL) * (self.frame_count / EPSILON_DECAY_LAST_FRAME)
            else:
                self.epsilon = EPSILON_FINAL
            
            self.frame_count += 1

            if ( np.random.random() < self.epsilon) :
                action = self.env.action_space.sample()
            else :
                q_values_tensor = self.network(state_tensor.unsqueeze(0))
                _, act_tensor = torch.max(q_values_tensor, dim=1)
                action = int(act_tensor.item())
            
            next_state , reward , is_done , is_trunc , _ = self.env.step(action)
            total_reward += reward
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            next_state_tensor = next_state_tensor.permute(2, 0, 1) 
            e = Experience(state_tensor,action,reward,is_done or is_trunc,next_state_tensor)
            self.replay_buffer.append(e)

            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32)
            state_tensor = state_tensor.permute(2, 0, 1)  

            if is_done or is_trunc :
                return total_reward 

    def train (self):

        optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.MSELoss()

        for epoch in range(30):

            for i in range(15):
                reward_ep  = self.play_episode()
                print(f"Episode reward: {reward_ep}, Epsilon: {self.epsilon:.4f}")

            dataset = ExperienceDataset(self.replay_buffer)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            for batch in dataloader:
                
                states = batch['state'].float()
                actions = batch['action'].long()
                rewards = batch['reward'].float()
                dones = batch['is_done_trunc'].bool()
                next_states = batch['next_state'].float()

                # Get current Q values for the taken actions
                current_q_values = self.network(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Calculate target Q values
                with torch.no_grad():
                    next_q_values = self.target_network(next_states)
                    max_next_q_values = next_q_values.max(1)[0]
                    
                    # Calculate targets: r + gamma * max(Q(s',a')) if not done, else just r
                    targets = rewards + GAMMA * max_next_q_values * (~dones)

                # Calculate loss
                loss = criterion(current_q_values, targets)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.target_network.load_state_dict(self.network.state_dict())
            print(f"Epoch {epoch}, Loss: {loss.item()}, Epsilon: {self.epsilon:.4f}")



if __name__ == "__main__" :
    gym.register_envs(ale_py)
    env = gym.make("BreakoutNoFrameskip-v4")
    a = Agent(env)
    a.train()
                
                




