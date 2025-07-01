import numpy as np
import torch 
import torch.nn as nn
import os
import glob
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MyCNN (nn.Module) :

   

    def __init__(self, input_channels, n_actions):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  
        
        self.relu = nn.ReLU()
        
        # Calculate the size after convolutions (for 96x96 input)
        # After conv1: (96-8)/4 + 1 = 23
        # After conv2: (23-4)/2 + 1 = 10  
        # After conv3: (10-3)/1 + 1 = 8
        # So final size is 8x8x64 = 4096
        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
  

    def forward(self, x):
        # Input shape: (batch_size, 4, 96, 96, 3) - 4 stacked frames
        # Need to reshape to (batch_size, channels, height, width)
        # Assuming input is (batch_size, 4, 96, 96, 3)
        batch_size = x.shape[0]
        
        # Reshape from (batch_size, 4, 96, 96, 3) to (batch_size, 12, 96, 96)
        # This concatenates the 4 frames along the channel dimension
        x = x.reshape(batch_size, -1, 96, 96)  # (batch_size, 12, 96, 96)
        
        # Convolutional layers with ReLU
        x = self.relu(self.conv1(x))  # (batch_size, 32, 23, 23)
        x = self.relu(self.conv2(x))  # (batch_size, 64, 10, 10)
        x = self.relu(self.conv3(x))  # (batch_size, 64, 8, 8)
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)  # (batch_size, 4096)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # (batch_size, 3) - [steering, gas, brake]
        
        # Apply tanh to steering (index 0) to keep it in [-1, 1]
        # Apply sigmoid to gas and brake (indices 1, 2) to keep them in [0, 1]
        steering = torch.tanh(x[:, 0:1])
        gas_brake = torch.sigmoid(x[:, 1:3])
        
        return torch.cat([steering, gas_brake], dim=1)

class TrajectoryDataset(Dataset):
    def __init__(self, states, actions):
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

def train_behavior_cloning(model, trajectory_dir, num_epochs=100, batch_size=32, learning_rate=1e-4, device='cuda'):
    """Train the model using behavior cloning on collected trajectories"""
    
    # Load all trajectory data
    states, actions = load_trajectories(trajectory_dir)
    
    # Create dataset and dataloader (use all data for training)
    train_dataset = TrajectoryDataset(states, actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
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
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'bc_model_epoch_{epoch+1}.pth')
            print(f"  Model saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_bc_model.pth')
    print("Training completed!")
    print("Final model saved as 'final_bc_model.pth'")
    return model

# Example usage:
if __name__ == "__main__":
    # Create model
    model = MyCNN(input_channels=12, n_actions=3)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_behavior_cloning(
        model=model,
        trajectory_dir="trajectories",  # Your trajectory directory
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        device=device
    )





