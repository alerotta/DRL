import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

class Discriminator (nn.Module):
    
    def __init__(self,input_channels_img = 3 ,input_channels_action = 3):

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels_img,32, kernel_size= 8 , stride= 4 )
        self.conv2 = nn.Conv2d(32,64, kernel_size= 4 , stride= 2 )
        self.conv3 = nn.Conv2d(64,64, kernel_size= 3 , stride= 1 )

        # Calculate correct output size: 96 -> 23 -> 10 -> 8
        out_conv_size = 8 * 8 * 64  # 4096

        self.lin = nn.Linear(input_channels_action, 128)

        self.fc1 = nn.Linear(out_conv_size + 128 , 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action ):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        action = F.relu(self.lin(action))

        #reshape
        x = x.view(x.size(0), - 1)
        combined = torch.cat((x,action), dim= 1 )

        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))

        #evaluate torch.sigmoid() as activation function. 
        combined = self.fc3(combined)

        return combined
    
    def reward (self,state,action):
        with torch.no_grad() :
            logits = self.forward(state,action)
            return -F.logsigmoid(-logits)






