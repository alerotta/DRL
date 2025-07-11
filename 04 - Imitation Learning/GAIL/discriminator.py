import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy


class Discriminator(nn.Module):


    def __init__(self, input_features_img, input_features_action, output):
        super(Discriminator, self).__init__()
        
        # Images - convolutional layers
        self.conv1 = nn.Conv2d(input_features_img, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened conv features
        # For 96x96 input: 96->23->11->9, so 9*9*64 = 5184
        self.conv_output_size = 8 * 8 * 64
        
        # Action processing
        self.fc_action = nn.Linear(input_features_action, 128)
        
        # Combined features (conv features + action features)
        self.fc1 = nn.Linear(self.conv_output_size + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        # Process the image (state) through conv layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the conv features
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, conv_output_size]
        
        # Process the action
        action_features = F.relu(self.fc_action(action))
        
        # Combine state and action features
        combined = torch.cat([x, action_features], dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply sigmoid to get probability
        return self.sigmoid(x)