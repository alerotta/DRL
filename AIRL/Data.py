from dataclasses import dataclass
from collections import deque 
from torch.utils.data import Dataset
import torch

@dataclass
class Step :
    state : torch.Tensor # stack 4 frames
    next_state : torch.Tensor #stack 4 frames
    action : torch.Tensor
    log_prob : torch.Tensor
    value : torch.Tensor
    reward : torch.Tensor
    done: bool
    advantage : torch.Tensor
    ret : torch.Tensor 
    expert: bool

class AIRLDataset (Dataset) :

    def __init__(self,steps) :
        self.steps = steps
        

    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, index):
        step = self.steps[index]
        return {
            'state': step.state,
            'next_state' : step.next_state,
            'action': step.action,
            'log_prob': step.log_prob,
            'value': step.value,
            'reward': step.reward,
            'done': step.done,
            'advantage': step.advantage,
            'ret': step.ret,
            'expert' : step.expert
        }
    