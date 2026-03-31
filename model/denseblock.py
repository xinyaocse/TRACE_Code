# -*- coding: utf-8 -*-
#model/denseblock.py
import torch
import torch.nn as nn
from .rrdb_denselayer import ResidualDenseBlock_out
import config as c

class Dense(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, inp, outp):
        super(Dense, self).__init__()
        self.dense = ResidualDenseBlock_out(inp, outp, nf=c.nf, gc=c.gc)

    def forward(self, x):
        return self.dense(x)


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
            out = self.inv2(x, rev=True)
            out = self.inv1(out, rev=True)
            return out
