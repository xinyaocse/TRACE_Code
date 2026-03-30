# -*- coding: utf-8 -*-
#model\model.py
import torch
import torch.nn as nn
from .hinet import Hinet_stage
import config as c

class Model(nn.Module):
    """HiNet_stage结构"""

    def __init__(self):
        super(Model, self).__init__()
        self.model = Hinet_stage()

    def forward(self, x, rev=False):
        if not rev:
            return self.model(x)
        else:
            return self.model(x, rev=True)

def init_model(mod):
    """初始化INN结构"""
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()


