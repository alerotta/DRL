import torch 
import torch.nn as nn
import torch.nn.functional as F

class AIRLReward (nn.Module):

    def __init__(self):
        super().__init__(self)

        self.conv1 = nn.Conv2d(12,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)

        conv_out_size = 8*8*64

        self.fc1 = nn.Linear(3,128)

        self.fc2 = nn.Linear(conv_out_size + 128 ,512)
        self.fc3 = nn.Linear(512 ,256)
        self.fc4 = nn.Linear(256 ,1)


    def forward (self,state,action) :

        state = state.view(state.size(0) , -1 , 96 ,96 ) 

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1)

        action =  F.relu(self.fc1(action))

        combined = torch.cat([x,action],dim=1)

        combined =  F.relu(self.fc2(combined))
        combined =  F.relu(self.fc3(combined))
        combined =  F.relu(self.fc4(combined))

        return combined
