import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Generator (nn.Module):

    def __init__(self,input_features,output_features):
        super().__init__()

        self.conv1 = nn.Conv2d(input_features,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)

        out_img_size = 8*8*64

        # Separate networks for mean and log_std
        # Mean network
        self.fc_mean1 = nn.Linear(out_img_size, 512)
        self.fc_mean2 = nn.Linear(512, 256)
        self.fc_mean3 = nn.Linear(256, output_features)
        
        # Log standard deviation network
        self.fc_logstd1 = nn.Linear(out_img_size, 512)
        self.fc_logstd2 = nn.Linear(512, 256)
        self.fc_logstd3 = nn.Linear(256, output_features)



    def forward(self, x):
        # Extract features using convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Compute mean of action distribution
        mean = F.relu(self.fc_mean1(x))
        mean = F.relu(self.fc_mean2(mean))
        mean = torch.tanh(self.fc_mean3(mean))  # Tanh to bound actions
        
        # Compute log standard deviation of action distribution
        log_std = F.relu(self.fc_logstd1(x))
        log_std = F.relu(self.fc_logstd2(log_std))
        log_std = self.fc_logstd3(log_std)
        
        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # Create normal distributions for each action dimension
        dist = Normal(mean, std)
        
        return dist
    
    def get_action(self, x):
        """Sample an action from the policy distribution"""
        dist = self.forward(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def get_log_prob(self, x, action):
        """Get log probability of a given action"""
        dist = self.forward(x)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob