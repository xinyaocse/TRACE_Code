# -*- coding: utf-8 -*-
#model/rrdb_denselayer.py
import torch
import torch.nn as nn
from modules.module_util import initialize_weights

class ResidualDenseBlock_out(nn.Module):
    """RRDB结构"""

    def __init__(self, inp, outp, nf=3, gc=32, bias=True, use_snorm=False):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(inp, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(inp+32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(inp+64, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(inp+96, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(inp+128, outp, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)

        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5

