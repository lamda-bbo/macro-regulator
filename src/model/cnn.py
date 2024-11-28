import numpy as np
import torch as th
import torch.nn as nn

from utils.debug import *

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 12, 1),
            nn.ReLU(),
            nn.Conv2d(12, 12, 1),
            nn.ReLU(),
            nn.Conv2d(12, 1, 1),
        )
    
    def forward(self, x):
        return self.cnn(x)
    
class MyCNNCoarse(nn.Module):
    def __init__(self, args, res_net):
        super(MyCNNCoarse, self).__init__()
        self.grid = args.grid
        self.devonv_grid = self.grid // 32
        self.cnn = res_net.to(args.device)
        self.cnn.conv1 = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.cnn.fc = nn.Linear(512, 16*self.devonv_grid*self.devonv_grid)
        self.devonv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #devonv_grid * 2
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #devonv_grid * 4
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #devonv_grid * 8
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #devonv_grid * 16
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #devonv_grid * 32
        )
    
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, self.devonv_grid, self.devonv_grid)
        return self.devonv(x), x

if __name__ == '__main__':
    pass