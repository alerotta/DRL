import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import os
import glob

class CarRacingCNN(nn.Module):

    def __init__(self,input_channels,output_actions):

        super().__init__() 
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size= 8 , stride= 4)
        self.conv2 = nn.Conv2d( 32,64,kernel_size= 4, stride= 2)
        self.conv3 = nn.Conv2d(64,64,kernel_size= 3, stride= 1)

        self.linear1 = nn.Linear (8*8*64,512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear (256,output_actions)

    def forward(self,x):

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 96, 96) 

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, -1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        steering = torch.tanh(x[:, 0:1])
        gas_brake = torch.sigmoid(x[:, 1:3])
        
        return torch.cat([steering, gas_brake], dim=1)
    

class TrajectoryDataset (Dataset):

    def __init__(self,states,actions):
        super().__init__()
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)   
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def load_trajectories(trajectory_dir):
    """Load all trajectory files from directory"""
    all_states = []
    all_actions = []
    
    # Find all .npz files in the directory
    trajectory_files = glob.glob(os.path.join(trajectory_dir, "*.npz"))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for file_path in trajectory_files:
        data = np.load(file_path)
        states = data['states']  # Shape: (n_steps, 4, 96, 96, 3)
        actions = data['actions']  # Shape: (n_steps, 3)
        
        # Normalize states to [0, 1]
        states = states.astype(np.float32) / 255.0
        
        all_states.append(states)
        all_actions.append(actions)
        print(f"Loaded {states.shape[0]} steps from {os.path.basename(file_path)}")
    
    # Concatenate all trajectories
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Total dataset: {all_states.shape[0]} state-action pairs")
    print(f"States shape: {all_states.shape}")
    print(f"Actions shape: {all_actions.shape}")
    
    return all_states, all_actions

def train_bc (model, trajectory_dir, num_epochs=2, batch_size=32, learning_rate=1e-4):

    states, actions = load_trajectories(trajectory_dir)

    train_dataset = TrajectoryDataset(states, actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    criterion = nn.MSELoss() #loss function 

    print(f"Training on {len(train_dataset)} samples")

    model.train() 
    for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_actions in train_loader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                predicted_actions = model(batch_states)
                
                # Calculate loss
                loss = criterion(predicted_actions, batch_actions)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for this epoch
            avg_loss = epoch_loss / num_batches
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
            
    model_path = 'trained_bc_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Training completed! Model saved as '{model_path}'")
    print("You can now load this model for testing.")
    return model

if __name__ == "__main__":
    model = CarRacingCNN(input_channels=12, output_actions=3)
    
    # Train the model
    trained_model = train_bc(
        model=model,
        trajectory_dir="trajectories",
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-4
    )

    

    






