import torch as th
import torch.nn as nn

from utils.debug import *
from utils.state_parsing import StateParsing

class Actor(nn.Module):
    def __init__(self, args, cnn, cnn_coarse) -> None:
        super(Actor, self).__init__()
        self.args = args
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.merge = nn.Conv2d(2, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.state_parsing = StateParsing(args)
        self.grid = args.grid

    def forward(self, x):
        # cnn_input contains all masks
        cnn_input = x[:, 1+self.grid *self.grid *2 : 1+self.grid *self.grid *8].reshape(-1, 6, self.args.grid, self.args.grid)
        position_mask = self.state_parsing.state2position_mask(x, next_next_macro=False)
        position_mask = position_mask.flatten(start_dim=1, end_dim=2)
        cnn_res = self.cnn(cnn_input)
        
        # coarse_input contains canvas, regular_mask and wire_mask
        old_canvas = self.state_parsing.state2canvas(x, new=False).unsqueeze(1)
        new_canvas = self.state_parsing.state2canvas(x, new=True).unsqueeze(1)
        regular_mask = self.state_parsing.state2regular_mask(x, next_next_macro=False).unsqueeze(1)
        next_regular_mask = self.state_parsing.state2regular_mask(x, next_next_macro=True).unsqueeze(1)
        wire_mask = self.state_parsing.state2wire_mask(x, next_next_macro=False).unsqueeze(1)
        next_wire_mask = self.state_parsing.state2wire_mask(x, next_next_macro=True).unsqueeze(1)
        coarse_input = th.cat([old_canvas, 
                               new_canvas, 
                               regular_mask, 
                               wire_mask, 
                               next_regular_mask, 
                               next_wire_mask],
                               dim=1)
        coarse_res, _ = self.cnn_coarse(coarse_input)
        cnn_res = self.merge(th.cat([cnn_res, coarse_res], dim=1))


        mask2 = self.args.wire_coeff * wire_mask.flatten(start_dim=1) + (1 - self.args.wire_coeff) * regular_mask.flatten(start_dim=1) + position_mask * 10
        mask_min = mask2.min() + self.args.soft_coefficient
        mask2 = mask2.le(mask_min).logical_not().float()

        x = cnn_res.reshape(-1, self.grid * self.grid)
        x = th.where(position_mask + mask2 >= 1.0, -1.0e10, x.double())
        x = self.softmax(x)
        return x
