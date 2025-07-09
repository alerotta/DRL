import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np

import gymnasium as gym
import pygame
from model import MyCnn 

import os
import glob

def load_trajectories(trajectory_dir):
    """Load all trajectory files from directory"""
    all_states = []
    all_actions = []
    
  
    trajectory_files = glob.glob(os.path.join(trajectory_dir, "*.npz"))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for file_path in trajectory_files:
        data = np.load(file_path)
        states = data['states']  
        actions = data['actions'] 
        
        
        states = states.astype(np.float32) / 255.0
        
        all_states.append(states)
        all_actions.append(actions)
        print(f"Loaded {states.shape[0]} steps from {os.path.basename(file_path)}")
    
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Total dataset: {all_states.shape[0]} state-action pairs")
    print(f"States shape: {all_states.shape}")
    print(f"Actions shape: {all_actions.shape}")
    
    return all_states, all_actions

class MyDataSet (Dataset) :

    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, indx):
        return self.states[indx] , self.actions[indx]


def train():

    #loading data
    states,actions = load_trajectories("DAgger/trajectories")
    data = MyDataSet (states=states, actions=actions)
    dloader = DataLoader(data, batch_size=32, shuffle=True )

    #model setup
    model = MyCnn(12,3)

    #optimizer 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(5):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_actions in dloader:
            batch_states = batch_states
            batch_actions = batch_actions
            
            # Forward pass
            predicted_actions = model(batch_states)
            loss = criterion(predicted_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        print(f"Epoch [{epoch+1}/{5}]")
        print(f"Train Loss: {avg_train_loss:.6f}")
        

    
    torch.save(model.state_dict(), 'final_bc_model.pth')
    print("Training completed!")
    print("Final model saved as 'final_bc_model.pth'")
    return model



if __name__ == "__main__" : 
    train()








    


