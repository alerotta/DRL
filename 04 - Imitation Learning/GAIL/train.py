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

def load_trajectories(trajectory_dir):
    """Load all trajectory files from directory"""
    all_states = []
    all_actions = []
    
    # Find all .npz files in the directory
    trajectory_files = glob.glob(os.path.join(trajectory_dir, "*.npz"))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for file_path in trajectory_files:
        data = np.load(file_path)

        all_states.append(data['states'])
        all_actions.append(data['actions'])
        print(f"Loaded steps from {os.path.basename(file_path)}")

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = all_states / 255.0
    
    return all_states, all_actions

    

class GAILDataset (Dataset) : 

    def __init__ (self, states, actions ,labels) : 
        super().__init__()

        assert len(states) == len(actions) == len(labels)
        
        # states shape: (batch_size, 4, 96, 96, 3) -> (batch_size, 4, 3, 96, 96)
        # Rearrange to: (batch, frames, channels, height, width) for discriminator
        self.states = torch.FloatTensor(states).permute(0, 1, 4, 2, 3)  
        self.actions = torch.FloatTensor(actions)
        self.labels = torch.FloatTensor(labels)


    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self.states[index],self.actions[index], self.labels[index]
    

def collect_r (model, env, episode_length): 
    obs, _ = env.reset()
    states = []
    actions = []

    
    for step in range(episode_length):
        # Get action from the generator (PPO policy)
        action, _ = model.predict(obs, deterministic=False)
        
        # Store the 4-frame stack and action
        # Get the current frame buffer (4 frames) from the environment
        frame_stack = env.get_frame_buffer() / 255.0  # Shape: (4, H, W, C)
        states.append(frame_stack.copy())
        actions.append(action.copy())
        
        # Take the action in the environment
        obs, _, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()

    # Convert to numpy arrays
    rollout_states = np.array(states)  # Shape: (episode_length, 4, H, W, C)
    rollout_actions = np.array(actions)

    return rollout_states,rollout_actions
    



def train_discriminator (discriminator,generator,w_env):

    
    states_exp, actions_exp  = load_trajectories("trajectories")
    states_ppo, actions_ppo = collect_r (generator, w_env,len(states_exp))
    labels_exp = torch.ones(len(states_exp), dtype=torch.float32)  
    labels_ppo = torch.zeros(len(states_ppo), dtype=torch.float32) 

    data_exp = GAILDataset(states_exp, actions_exp, labels_exp)
    data_ppo = GAILDataset(states_ppo, actions_ppo, labels_ppo)

    combined_dataset = ConcatDataset([data_exp, data_ppo])
    dataloader = DataLoader(combined_dataset, 64, True)
    
    # Set up discriminator training
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0003)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop for discriminator
    num_epochs = 5
    discriminator.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (states , actions , labels) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = discriminator(states,actions)
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


def train():

    discriminator = Discriminator(12,3)
    w_env = GAILWrapper(gym.make("CarRacing-v3"),discriminator)
    generator = PPO("MlpPolicy",w_env)

    for _ in range(5):
     generator.learn(total_timesteps=400)
     train_discriminator(discriminator,generator,w_env)
    
    # Save the trained PPO model
    generator.save("gail_ppo_model")
    print("PPO model saved as 'gail_ppo_model'")
    

    return 

if __name__ == "__main__" : 
    train()








