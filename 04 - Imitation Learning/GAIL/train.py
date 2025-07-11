import numpy as np
import glob
import os

import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
from collections import deque

from torch.utils.data import Dataset,DataLoader
from discriminator import Discriminator
from generator import Generator
from value_net import ValueNet


def load_from_dir (trajectory_dir):

    files = glob.glob(os.path.join(trajectory_dir,"*.npz"))

    all_states = []
    all_actions = []

    for file in files :
        data = np.load(file)
        all_states.append(data['states'])
        all_actions.append(data['actions'])

        print (f"loaded data from {os.path.basename(file)}")

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = all_states / 255.0

    return all_states,all_actions

class MyDataset(Dataset):

    def __init__(self,states,actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]
    
def collect_trajectories(env, generator_net, num_steps):
    """Collect trajectories using current policy"""
    states = []
    actions = []
    
    obs, _ = env.reset()
    frame_buffer = deque(maxlen=4)
    
    # Initialize frame buffer
    for _ in range(4):
        frame_buffer.append(obs)
    
    step_count = 0
    
    while step_count < num_steps:
        # Prepare state (4 stacked frames)
        state = np.stack(frame_buffer, axis=0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0) / 255.0
        
        # Get action from generator
        with torch.no_grad():
            action, _ = generator_net.get_action(state_tensor)
            action = action.squeeze().cpu().numpy()
        
        # Store state-action pair
        states.append(state)
        actions.append(action)
        
        # Take step in environment
        obs, reward, done, truncated, _ = env.step(action)
        frame_buffer.append(obs)
        
        step_count += 1
        
        if done or truncated:
            obs, _ = env.reset()
            frame_buffer.clear()
            for _ in range(4):
                frame_buffer.append(obs)
    
    return np.array(states), np.array(actions)


def train_discriminator(discriminator, optimizer, expert_dataloader, 
                       policy_states, policy_actions, num_epochs):
    """Train discriminator to distinguish expert from policy trajectories"""
    discriminator.train()
    criterion = nn.BCELoss()
    
    # Convert policy data to tensors
    policy_states_tensor = torch.FloatTensor(policy_states) / 255.0
    policy_actions_tensor = torch.FloatTensor(policy_actions)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for expert_states, expert_actions in expert_dataloader:
            batch_size = expert_states.size(0)
            
            # Sample policy data
            indices = torch.randint(0, len(policy_states_tensor), (batch_size,))
            policy_batch_states = policy_states_tensor[indices]
            policy_batch_actions = policy_actions_tensor[indices]
            
            # Forward pass
            expert_pred = discriminator(expert_states, expert_actions)
            policy_pred = discriminator(policy_batch_states, policy_batch_actions)
            
            # Labels: 1 for expert, 0 for policy
            expert_labels = torch.ones_like(expert_pred)
            policy_labels = torch.zeros_like(policy_pred)
            
            # Compute loss
            expert_loss = criterion(expert_pred, expert_labels)
            policy_loss = criterion(policy_pred, policy_labels)
            loss = expert_loss + policy_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if epoch % 2 == 0:
            print(f"  Discriminator epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def train_policy(generator, value_net, discriminator, gen_optimizer, 
                value_optimizer, states, actions, num_epochs):
    """Train policy using discriminator-generated rewards"""
    generator.train()
    value_net.train()
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states) / 255.0
    actions_tensor = torch.FloatTensor(actions)
    
    # Generate rewards from discriminator
    with torch.no_grad():
        discriminator_outputs = discriminator(states_tensor, actions_tensor)
        rewards = -torch.log(discriminator_outputs + 1e-8)  # Add small epsilon for stability
    
    # Compute advantages (simplified - you might want to use GAE)
    with torch.no_grad():
        values = value_net(states_tensor)
        advantages = rewards - values
        returns = rewards
    
    for epoch in range(num_epochs):
        # Update value network
        value_optimizer.zero_grad()
        predicted_values = value_net(states_tensor)
        value_loss = nn.MSELoss()(predicted_values, returns)
        value_loss.backward()
        value_optimizer.step()
        
        # Update generator
        gen_optimizer.zero_grad()
        log_probs = generator.get_log_prob(states_tensor, actions_tensor)
        policy_loss = -(log_probs * advantages.detach()).mean()
        policy_loss.backward()
        gen_optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Policy epoch {epoch+1}/{num_epochs}, "
                  f"Value Loss: {value_loss.item():.4f}, "
                  f"Policy Loss: {policy_loss.item():.4f}")


def evaluate_policy(env, generator, num_episodes=5):
    """Evaluate current policy performance"""
    generator.eval()
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        frame_buffer = deque(maxlen=4)
        
        # Initialize frame buffer
        for _ in range(4):
            frame_buffer.append(obs)
        
        episode_reward = 0
        done = False
        
        while not done:
            state = np.stack(frame_buffer, axis=0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                action, _ = generator.get_action(state_tensor)
                action = action.squeeze().cpu().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            frame_buffer.append(obs)
            episode_reward += reward
            
            if truncated:
                done = True
        
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"  Evaluation: Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    generator.train()


def train ():
    # Load expert demonstrations
    states, actions = load_from_dir("trajectories")
    expert_dataset = MyDataset(states, actions)
    expert_dataloader = DataLoader(expert_dataset, batch_size=64, shuffle=True)

    # Initialize networks
    input_channels = 4  # 4 stacked frames
    action_dim = 3  # CarRacing action space
    
    value_net = ValueNet(input_channels)
    discriminator_net = Discriminator(input_channels, action_dim, 1)
    generator_net = Generator(input_channels, action_dim)
    
    # Initialize optimizers
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=3e-4)
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=3e-4)
    
    # Initialize environment
    env = gym.make("CarRacing-v3", render_mode=None)
    
    # Training hyperparameters
    num_epochs = 1000
    collect_steps = 2000  # Steps to collect per epoch
    discriminator_epochs = 5
    policy_epochs = 10
    
    print(f"Expert dataset size: {len(expert_dataset)}")
    print("Starting GAIL training...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # Phase 1: Collect policy trajectories
        print("Collecting policy trajectories...")
        policy_states, policy_actions = collect_trajectories(
            env, generator_net, collect_steps
        )
        
        # Phase 2: Train discriminator
        print("Training discriminator...")
        train_discriminator(
            discriminator_net, discriminator_optimizer,
            expert_dataloader, policy_states, policy_actions,
            discriminator_epochs
        )
        
        # Phase 3: Train policy using discriminator rewards
        print("Training policy...")
        train_policy(
            generator_net, value_net, discriminator_net,
            generator_optimizer, value_optimizer,
            policy_states, policy_actions, policy_epochs
        )
        
        # Evaluation and logging
        if epoch % 10 == 0:
            evaluate_policy(env, generator_net, num_episodes=5)
    
    env.close()
    print("Training completed!")


if __name__ == "__main__":
    train()


