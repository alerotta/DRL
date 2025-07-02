import torch 
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNNFeatureExtractor

class ValueNetwork (nn.Module):
    
    def __init__(self, feature_dim= 512, hidden_dim= 265):
        super(ValueNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels=12,output_features=feature_dim)

        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim,hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim,hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim,1)

        )

    def forward (self,state): 
        features = self.cnn(state)
        value = self.mean_head(features)
        return value

        
