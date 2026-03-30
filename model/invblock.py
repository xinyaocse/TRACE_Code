# -*- coding: utf-8 -*-
#model/invblock.py
import torch
import torch.nn as nn
from .denseblock import Dense
import config as c

class INV_block_affine(nn.Module):
    """
    可逆信息交换关键结构 (ψ, φ, ρ, η) × clamp
    """

    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp

        # ρ, η, φ, ψ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        self.p = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        # clamp * 2 * (sigmoid(s) - 0.5)
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=True):
        """
        正向：x -> y
        rev=True 时可实现逆过程
        """
        if not rev:
            x1 = x[:, :self.split_len1, :, :]
            x2 = x[:, self.split_len1:self.split_len1+self.split_len2, :, :]

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1 = self.r(y1)
            t1 = self.y(y1)
            y2 = self.e(s1) * x2 + t1
            return torch.cat([y1, y2], dim=1)
        else:
            # 若需要可逆逆过程, 这里可实现
            raise NotImplementedError("Reverse pass not implemented here.")

