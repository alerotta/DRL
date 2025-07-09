
import torch
import torch.nn as nn
import torch.nn.functional as F

class CarRacingCNN(nn.Module):

    def __init__(self,input_channels,output_actions):

        super().__init__() 
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size= 8 , stride= 4)
        self.conv2 = nn.Conv2d( 32,64,kernel_size= 4, stride= 2)
        self.conv3 = nn.Conv2d(64,64,kernel_size= 3, stride= 1)

        self.linear1 = nn.Linear (8*8*64,512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear (256,output_actions)

    def forward(self,x):

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 96, 96) 

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, -1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        steering = torch.tanh(x[:, 0:1])
        gas_brake = torch.sigmoid(x[:, 1:3])
        
        return torch.cat([steering, gas_brake], dim=1)
    