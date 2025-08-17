import torch 
import torch.nn as nn

#input featuresis the number of elements of the state, it is the dimension of the input tensor.

class MultiHeadNetwork (nn.Module):

    def __init__(self,input_features , n_actions):
        super().__init__()

        self.shared = nn.Sequential (
            nn.Linear(input_features,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128,n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(128,1)
        )
    
    def forward(self,x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value 