import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNFeatureExtractor(nn.Module):

    def __init__(self, input_channels=12, output_features=512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size= 8 , stride= 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 4 , stride= 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size= 3 , stride= 1)

        self.conv_out_size = self._get_conv_out_size(input_channels,96,96)

        self.fc1 = nn.Linear(self.conv_out_size,output_features)

    def _get_conv_out_size (self,channels,h,w):

        x = torch.zeros(1,channels,h,w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return int(np.prod(x.size()[1:])) 

    def forward (self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))

        return x


