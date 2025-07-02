import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNNFeatureExtractor
from torch.distributions import Normal

class PolicyNetwork (nn.Module) :

    def __init__(self, feature_dim = 512, action_dim = 3, hidden_dim = 256 ):
        super(PolicyNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels=12,output_features=feature_dim)

        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim,hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim,hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim,action_dim),
            F.relu(),
        )

        self.log_std_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward (self, state):

        features = self.cnn(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        #last activation 
        steering = torch.tanh(mean[:, 0:1])
        gas_brake = torch.sigmoid(mean[:, 1:3])
        mean = torch.cat([steering, gas_brake], dim=1)

        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean,std
    
    def get_action(self, state):
        with torch.no_grad():
            mean,std = self.forward(state)
            dist = Normal(mean,std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1) #needed for PPO

        return action.cpu().numpy() , log_prob.item()
    
    def evaluate_action(self, state, action):
        mean, std = self.forward(state)
        dist = Normal( mean,std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy




 
    
    
