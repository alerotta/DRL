import torch 
import torch.nn as nn 
import torch.nn.functional as F
from reward import AIRLReward

class AIRLDiscriminator (nn.Module):
    
    def __init__(self,reward_network : AIRLReward, policy_network):
        super().__init__()
        self.reward_network = reward_network
        self.policy_network = policy_network

    def get_logits (self,state,action):

        r = self.reward_network(state,action)
        log_pi = self.policy.get_log_prob(state, action)
        logits = r - log_pi
        return logits
