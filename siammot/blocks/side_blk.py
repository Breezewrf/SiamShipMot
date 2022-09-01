from typing_extensions import Self
import torch
from torch import nn 
class Side(nn.Module):
    def __init__(self, out_ch):
        super(Side, self).__init__()
        self.out_channels = out_ch
        self.block = nn.ModuleDict({
            "side_0":nn.Conv2d(64, self.out_channels, 4, padding=1),
            "side_1":nn.Conv2d(64, self.out_channels, 3, padding=1),
            "side_2":nn.Conv2d(64, self.out_channels, 3, padding=1),
            "side_3":nn.Conv2d(64, self.out_channels, 3, padding=1),
            "side_4":nn.Conv2d(64, self.out_channels, 3, padding=1),
            "side_5":nn.Conv2d(64, self.out_channels, 3, padding=1),
            "side_6":nn.Conv2d(64, self.out_channels, 3, padding=1),
        }
        )
    
    def forward(self, x):
        return self.block["side_0"](x[0]), self.block["side_1"](x[1]), self.block["side_2"](x[2]), self.block["side_3"](x[3])