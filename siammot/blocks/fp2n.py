# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 17:03
# @Author  : Breeze
# @Email   : 578450426@qq.com
from siammot.blocks.RSU import *
from siammot.blocks.RSU import _upsample_like
from torch import sigmoid, tanh, nn


class fp2n(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, is_mask_decoder=True):
        super(fp2n, self).__init__()
        self.level_1 = RSU4F(512, 128, 256)
        self.level_2 = RSU4(512, 128, 256)
        self.level_3 = RSU5(384, 64, 128)
        self.level_4 = RSU6(192, 64, 128)

        self.side_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.side_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.side_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.side_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        
    def init_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("fp2n init finished!\n")    

    # 加入encode的跳跃连接
    def forward(self, x):
        top = x[3]

        d1 = self.level_1(top)
        d1_ = f.interpolate(d1, scale_factor=2, mode="nearest")
        # d1_ = _upsample_like(d1, x[2])

        d2 = self.level_2(torch.cat((d1_, x[2]), 1))
        d2_ = f.interpolate(d2, scale_factor=2, mode="nearest")
        # d2_ = _upsample_like(d2, x[1])

        d3 = self.level_3(torch.cat((d2_, x[1]), 1))
        d3_ = f.interpolate(d3, scale_factor=2, mode="nearest")
        # d3_ = _upsample_like(d3, x[0])

        d4 = self.level_4(torch.cat((d3_, x[0]), 1))

        return tanh(self.side_4(d4)), tanh(self.side_3(d3)), tanh(self.side_2(d2)), tanh(self.side_1(d1))
    # return tanh(d0), tanh(d1), tanh(d2), tanh(d3), tanh(d4), tanh(d5),
    #
    # return sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5)


# v1
class fp2n_add(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, is_mask_decoder=True):
        super(fp2n_add, self).__init__()
        self.level_1 = RSU4F(512, 128, 256)
        self.level_2 = RSU4(256, 64, 128)
        self.level_3 = RSU5(128, 32, 64)
        self.level_4 = RSU6(64, 32, 64)

        self.side_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.side_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.side_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.side_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        
    def init_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("fp2n init finished!\n")    

    # 加入encode的跳跃连接
    def forward(self, x):
        top = x[3]

        d1 = self.level_1(top)
        d1_ = f.interpolate(d1, scale_factor=2, mode="bilinear")
        # d1_ = _upsample_like(d1, x[2])
        d2 = self.level_2(torch.add(d1_, x[2]))
        d2_ = f.interpolate(d2, scale_factor=2, mode="bilinear")
        # d2_ = _upsample_like(d2, x[1])

        d3 = self.level_3(torch.add(d2_, x[1]))
        d3_ = f.interpolate(d3, scale_factor=2, mode="bilinear")
        # d3_ = _upsample_like(d3, x[0])

        d4 = self.level_4(torch.add(d3_, x[0]))

        return tanh(self.side_4(d4)), tanh(self.side_3(d3)), tanh(self.side_2(d2)), tanh(self.side_1(d1))
        # return tanh(d4), tanh(d3), tanh(d2), tanh(d1)