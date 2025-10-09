import torch
import torch.optim as optim
from torch.utils.data import DataLoader 
import torch.nn.functional  as F

import Network as net
import numpy as np

from Data import Step , AIRLDataset

import os
import glob

class Discriminator :

    def __init__(self,state_dim,action_dim,expert_path,state_encoder, hidden_size = 256, lr=3e-4):

        self.network = net.DiscNetwork(state_dim,action_dim,hidden_size,state_encoder)
        self.opt = optim.Adam(self.network.parameters(), lr =lr)
        self.expert_path = expert_path
    
    def _load_expert_trajectories (self ,path):

        trajectory_files = glob.glob(os.path.join(path, "*.npz"))
        steps = []
        all_states = []
        all_actions = []

        for file_path in trajectory_files:

            data = np.load(file_path)
            all_states.append(data['states'])
            all_actions.append(data['actions'])
            #convert to torch tensors
        
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        for i in range (len(all_states)):
            if i == len(all_states) -1 :
                continue
            
            s = Step(
                state=torch.tensor(all_states[i], dtype=torch.float32) / 255.0,
                next_state=torch.tensor(all_states[i+1], dtype=torch.float32) / 255.0,
                action=torch.tensor(all_actions[i], dtype=torch.float32),
                log_prob=torch.zeros(1),
                value=torch.zeros(1),
                reward=torch.zeros(1),
                done=False,
                advantage=torch.zeros(1),
                ret= torch.zeros(1),
                expert= True
            )

            steps.append(s)
        
        return steps 
    
    def get_discriminator_dataset (self, steps_from_ppo , expert_path):

        expert_steps = self._load_expert_trajectories(expert_path)
        steps = steps_from_ppo + expert_steps
        return AIRLDataset(steps=steps)
    
    def update_discriminator(self, steps_from_ppo, epochs=4, batch_size=64):
        dataset = self.get_discriminator_dataset(steps_from_ppo, self.expert_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        avg_loss, n_batches = 0.0, 0
        for _ in range(epochs):
            for batch in dataloader:
                states  = batch['state']
                next_states = batch['next_state']
                actions = batch['action']
                expert  = batch['expert']

                # Prefer recomputing log π(a|s) with current policy; if not available, keep stored:
                log_probs = batch['log_prob']

                f_vals, _, _, _ = self.network(states, actions, next_states)
                logits = f_vals - (1.0 - expert) * log_probs

                loss = F.binary_cross_entropy_with_logits(logits, expert)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                self.opt.step()

                avg_loss += loss.item()
                n_batches += 1
        return avg_loss / max(n_batches, 1)

        
    def reward (self, s , a , sp):
        return self.network.reward(s,a,sp)

    
