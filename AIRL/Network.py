import torch
import torch.nn as nn
from torch.distributions import Normal

class CNNEncoder (nn.Module):
    def __init__(self, state_dim , hidden_size = 256 ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[-1]*4, 32 , kernel_size=8 , stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64 , kernel_size=4 , stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64 , kernel_size=3 , stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.conv_out_dim = self.get_conv_out_dim(state_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
    
    def get_conv_out_dim (self, state_dim):
        dummy = torch.zeros(1, state_dim[2]* 4,state_dim[0],state_dim[1])
        x = self.conv(dummy)
        return x.numel()
    
    def forward (self, x ):
        z = self.conv(x)
        z = self.fc(z)
        return z


class PPONetwork (nn.Module):


    def __init__(self, state_dim , action_dim   , hidden_size = 256 ,shared_encoder = None):
        super().__init__()

        # state dim is [H ,W ,C]

        self.state_dim = state_dim
        self.action_dim = action_dim

        if shared_encoder is not None :
            self.convolutional = shared_encoder
        else:
            self.convolutional = CNNEncoder(state_dim, hidden_size)

        self.continuous_means = nn.Linear(hidden_size, action_dim)
        self.continuous_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5) 
        self.value_head = nn.Linear(hidden_size , 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)


    
    def forward (self,x) :

        distributions = []

        
        shared_features = self.convolutional(x)
        means = self.continuous_means(shared_features)
        stds = torch.exp(self.continuous_log_std)
        value = self.value_head(shared_features)

        for i in range(self.action_dim):
            continuous_dist = Normal(means[:, i], stds[i])
            distributions.append(continuous_dist)

        return distributions , value
    
    def sample_action (self, state):

        distributions , value  = self.forward(state)

        actions = []
        log_probs = []

        for i,dist in enumerate(distributions):

            action = dist.sample()
            log_prob = dist.log_prob(action)

            if i == 0:
                action = torch.clamp(action,-1,1)
            else:
                action = torch.clamp(action,0,1)

            

            actions.append(action.unsqueeze(-1)) 
            log_probs.append(log_prob.unsqueeze(-1)) 

        combined_action = torch.cat(actions, dim=-1)
        combined_log_prob = torch.cat(log_probs, dim=-1).sum(dim=-1, keepdim=True)

        return combined_action , combined_log_prob , value
    
    def evaluate_actions(self, state, actions):
        """
        Evaluate actions under the current policy.
        This is crucial for PPO: we need to evaluate OLD actions under the NEW policy.
        """
        distributions, value = self.forward(state)
        
        log_probs = []
        entropy = []
        
        for i, dist in enumerate(distributions):
            # Calculate log probability of the given action
            log_prob = dist.log_prob(actions[:, i])
            log_probs.append(log_prob.unsqueeze(-1))
            
            # Calculate entropy for this dimension
            entropy.append(dist.entropy())
        
        combined_log_prob = torch.cat(log_probs, dim=-1).sum(dim=-1, keepdim=True)
        total_entropy = torch.stack(entropy).sum(dim=0).mean()
        
        return combined_log_prob, value, total_entropy
        

class DiscNetwork (nn.Module):

    def __init__(self, state_dim , action_dim  , hidden_size = 256 , shared_encoder = None):

        super().__init__()
        if shared_encoder is not None :
            self.convolutional = shared_encoder
        else:
            self.convolutional = CNNEncoder(state_dim, hidden_size)
        
        half = int(hidden_size // 2)

        self.g = nn.Sequential(
            nn.Linear(hidden_size + action_dim , half),
            nn.ReLU(),
            nn.Linear(half , 1)
        )

        self.h = nn.Sequential(
            nn.Linear(hidden_size , half),
            nn.ReLU(),
            nn.Linear(half , 1)
        )

        self.gamma = 0.99
    
    def forward(self, s, a, sp):
        z  = self.convolutional(s)
        zp = self.convolutional(sp)
        g_out = self.g(torch.cat([z, a], dim=1)).squeeze(-1)
        h_s   = self.h(z).squeeze(-1)
        h_sp  = self.h(zp).squeeze(-1)   
        f = g_out + self.gamma * h_sp - h_s
        return f, g_out, h_sp, h_s
    
    @torch.no_grad()
    def reward(self, s, a, sp):
        f, _, _, _ = self.forward(s, a, sp)
        return f.detach()
    

          



