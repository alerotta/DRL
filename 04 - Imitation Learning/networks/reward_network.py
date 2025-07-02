import torch 
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNNFeatureExtractor

class RewardNetwork(nn.Module):
    def __init__(self, feature_dim= 512, action_dim = 3, hidden_dim=256):
        super(RewardNetwork,self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels=12, output_features=feature_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward (self,state,action):
        features = self.cnn(state)
        combined = torch.cat([features, action], dim=1)
        reward = self.reward_head(combined)
        return reward


