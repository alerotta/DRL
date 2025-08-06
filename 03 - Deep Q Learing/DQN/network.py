import torch
import torch.nn as nn

class CnnNetwork (nn.Module):

    def __init__(self,input_shape , n_actions):

        #input shape of a image [x,y,3]

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64,kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Create dummy input with correct shape: (batch, channels, height, width)
        # input_shape is (210, 160, 3), so we need (1, 3, 210, 160)
        dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        conv_out_dim = self.conv(dummy_input).size()[-1]

        self.linear = nn.Sequential(
            nn.Linear(conv_out_dim,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,n_actions)
        )

    def forward(self, x: torch.tensor):
        x = x / 255.0
        return self.linear(self.conv(x))