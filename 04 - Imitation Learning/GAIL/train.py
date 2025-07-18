import numpy as np 
import gymnasium as gym
import torch
import os 
import glob

from torch.utils.data import Dataset,DataLoader,ConcatDataset
from collections import deque
from stable_baselines3 import PPO
from environment import EnvWrapper
from discriminator import Discriminator


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
    
    return all_states, all_actions

    

class GAILDataset (Dataset) : 

    def __init__ (self, states, actions ,labels, already_t : bool) : 
        super().__init__()
        assert len(states) == len(actions) == len(labels)

        if already_t :
            self.states = states
            self.actions = actions
            self.labels = labels
        else: 
            self.states = torch.FloatTensor(states)
            self.states = self.states.view(self.states.size(0) , -1 , 96 ,96 )
            self.states = self.states / 255.0
            self.actions = torch.FloatTensor(actions)
            self.labels = labels
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self.states[index],self.actions[index], self.labels[index]
    

def train_disc(discriminator: Discriminator, policy: PPO, env: EnvWrapper):

    #collect expert data
    trajectory_dir = "trajectories"
    expert_states, expert_actions = load_trajectories(trajectory_dir)
    expert_labels = torch.ones(len(expert_actions))
    dataset_expert = GAILDataset(expert_states, expert_actions, expert_labels, False)
        
    # Collect policy trajectories
    policy_states = []
    policy_actions = []
        
    obsv, info = env.reset()
    
    print(f"generating: {len(expert_actions)} expert actions")
    for i in range(len(expert_actions)):
        # Get action from PPO policy
        action, _ = policy.predict(obsv, deterministic=False)
        state_tensor = env.get_state_tensor()
        action_tensor = torch.FloatTensor(action).unsqueeze(0)

        # Append state and action tensors to policy data
        policy_states.append(state_tensor)
        policy_actions.append(action_tensor)

            
        obsv, reward, is_terminated, is_truncated, info = env.step(action)


            
        # Reset if episode ends
        if is_terminated or is_truncated:
            print(f"generated {i} out of {len(expert_actions)} expert_actions")
            obsv, info = env.reset()
        
    # Convert lists to tensors and create labels
    policy_states = torch.cat(policy_states, dim=0)
    policy_actions = torch.cat(policy_actions, dim=0)
    policy_labels = torch.zeros(len(policy_actions))
        
    dataset_policy = GAILDataset(policy_states, policy_actions, policy_labels, True)

    # Concatenate datasets
    dataset_combined = ConcatDataset([dataset_expert, dataset_policy])
    
    # Create DataLoader
    dataloader = DataLoader(dataset_combined, batch_size=32, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    epochs = 2
    print(f"generated {len(expert_actions)} expert_actions")

    for epoch in range(epochs):
        print (f"epoch: {epoch}")
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        disc.train()
        
        for batch_s , batch_a , batch_l  in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = discriminator(batch_s, batch_a)
            logits = logits.squeeze()  # Remove extra dimensions
            
            # Calculate loss
            loss = criterion(logits, batch_l)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
    
    return 0




    

if __name__ == "__main__" : 

    disc = Discriminator()
    real_env  = gym.make("CarRacing-v3")
    w_env = EnvWrapper(real_env,disc)
    
    # Configure PPO for faster training
    policy = PPO("MlpPolicy", w_env, )

    for i in range(5):
        train_disc(disc,policy,w_env)
        print(f"disc learning step: {i} done!")
        policy.learn(total_timesteps=3000)
        print(f"PPO learning step: {i} done!")

    policy.save("gail_ppo_model")
    print("PPO model saved as 'gail_ppo_model'")


    