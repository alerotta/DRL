import torch 
import torch.nn as nn

class MultiHeadNetwork (nn.Module) :

    def __init__(self,obs_dim = 8  , n_actions = 4):
        super().__init__()
        self.common_body = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 1),  
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(64,32),  
            nn.ReLU(), 
            nn.Linear(32,n_actions),  
        )

    def forward(self, state):
        features = self.common_body(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    

if __name__ == "__main__" :

    net = MultiHeadNetwork(8,4)
    x  = torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float32)
    
    l , val = net(x)
    dist = torch.distributions.Categorical(logits=l)
    a_t = dist.sample()
    action = int(a_t.item())
    print(action)




   
