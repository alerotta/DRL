import torch 
import torch.nn as nn

class MultiHeadNetwork (nn.Module) :

    def __init__(self,obs_dim, n_actions):
        super().__init__()
        self.common_body = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),      
            nn.ReLU(),   
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 1)  
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(64, n_actions)
        )

    def forward(self, state):
        features = self.common_body(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)

   
