import numpy as np
import gymnasium as gym 

import torch
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torch.utils.data import ConcatDataset

from environment import GAILWrapper
from discriminator import Discriminator
from stable_baselines3 import PPO

import os 
import glob

def load_trajectories (trajectory_path):

    files = glob.glob(os.path.join(trajectory_path,"*.npz"))
    
    all_states = []
    all_actions = []
    
    for file in files :
        print(f"loading data from {file}")
        data = np.load(file)
        all_states.append(data['states'])
        all_actions.append(data['actions'])


    all_states = np.concatenate(all_states, axis=0)
    all_states = all_states / 255.0 #normalize
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"loaded {len(all_states)} state, action pairs.")
    
    return all_states,all_actions

class GAILDataset (Dataset) : 

    def __init__ (self, states, actions ,labels) : 
        super().__init__()
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.pairs = torch.cat((self.states,self.actions),dim=1)

        self.labels = labels

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self.pairs[index], self.labels[index]
    

def collect_r (model, env): 
    obs, _ = env.reset()
    states = []
    actions = []

    episode_length = 1000  # or however long you want the episode
    for step in range(episode_length):
        # Get action from the generator (PPO policy)
        action, _ = model.predict(obs, deterministic=False)
        
        # Store the state and action
        states.append(obs.copy()/255.0)
        actions.append(action.copy())
        
        # Take the action in the environment
        obs, _, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()

    # Convert to numpy arrays
    rollout_states = np.array(states)
    rollout_actions = np.array(actions)

    return rollout_states,rollout_actions
    



def train_discriminator ():

    states_exp, actions_exp  = load_trajectories("trajectories")
    
    discriminator = Discriminator(3,3)
    w_env = GAILWrapper(gym.make("CarRacing-v3"),discriminator)
    generator = PPO("MlpPolicy",w_env)

    states_exp, actions_exp  = load_trajectories("trajectories")
    states_ppo, actions_ppo = collect_r (generator, w_env)
    labels_exp = torch.ones(len(states_exp))  
    labels_ppo = torch.zeros(len(states_ppo)) 

    data_exp = GAILDataset(states_exp, actions_exp, labels_exp)
    data_ppo = GAILDataset(states_ppo, actions_ppo, labels_ppo)

    combined_dataset = ConcatDataset([data_exp, data_ppo])
    dataloader = DataLoader(combined_dataset, 64, True)
    
    # Set up discriminator training
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0003)
    criterion = torch.nn.BCELoss()
    
    # Training loop for discriminator
    num_epochs = 100
    discriminator.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (state_action_pairs, labels) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = discriminator(state_action_pairs)
            predictions = predictions.squeeze()  # Remove extra dimensions
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Optional: Save discriminator checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
            print(f"Saved discriminator checkpoint at epoch {epoch+1}")
    
    print("Discriminator training completed!")
    return discriminator, generator




