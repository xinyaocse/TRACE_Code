#config.py
# -*- coding: utf-8 -*-
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 学习率设置
lr = 2e-3  # TRACE攻击的学习率
iae_lr = 3e-3  # IAE的学习率

# 扰动限制
IAE_eps = 8.0/255.0  # IAE增强的扰动限制
eps = 8.0/255.0      # TRACE攻击的扰动限制

# INN网络参数
nf = 3
gc = 32
clamp = 2.0

# 损失函数权重
lamda_per = 12.0             # 特征相似性权重（从3.0提升到12.0）
lamda_low_frequency = 2.0    # 低频保持权重
lamda_j_default = 0.5        # H度量默认权重（从0.3提升到0.5）

# 初始化尺度
init_scale = 0.01

# 其他参数
m = 50  # 目标样本数量
num_targets_rie = 4  # RIE中使用的目标数量

# 黑盒查询参数
default_z = 4  # 从4提升到7
default_k = 10  # 从15提升到30

# Hash二值化
use_binary_hash = True  # 启用二值化哈希
