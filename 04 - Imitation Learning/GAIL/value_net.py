import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    
    def __init__(self, input_features):
        super().__init__()

        self.conv1 = nn.Conv2d(input_features, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        out_img_size = 8 * 8 * 64

        self.fc1 = nn.Linear(out_img_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the features
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer - values can be negative

        return x