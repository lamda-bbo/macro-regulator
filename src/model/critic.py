import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.debug import *
from utils.state_parsing import StateParsing

class Critic(nn.Module):
    def __init__(self, args, cnn_coarse):
        super(Critic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)

        self.net = nn.Sequential(
            nn.Linear(786, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.cnn_coarse = cnn_coarse

        self.state_parsing = StateParsing(args=args)

    def forward(self, x):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value
