# -*- coding: utf-8 -*-
#model/hinet.py
import torch.nn as nn
from .invblock import INV_block_affine

class Hinet_stage(nn.Module):
    """HiNet(INV_block_affine) composed of two affine invertible blocks"""

    def __init__(self):
        super(Hinet_stage, self).__init__()
        self.inv1 = INV_block_affine()
        self.inv2 = INV_block_affine()

    def forward(self, x, rev=False):
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            return out
        else:
            # 反向过程
            out = self.inv2(x, rev=True)
            out = self.inv1(out, rev=True)
            return out

