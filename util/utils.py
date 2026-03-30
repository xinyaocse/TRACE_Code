#util/utils.py

# -*- coding: utf-8 -*-

import os
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch, random, numpy as np
def load_image(imgpath):
    """加载单张图像并转换为 [1,3,224,224] 张量"""
    image = Image.open(imgpath).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def guide_loss(output, target):
    """计算引导损失，使用MSE"""
    return F.mse_loss(output, target)

def clamp(x, low, high):
    """将张量值限制在[low, high]范围内"""
    return torch.clamp(x, min=low, max=high)

def dcg(scores, k):
    """计算DCG (Discounted Cumulative Gain)"""
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

def ndcg(scores, ideal_scores, k):
    """计算NDCG (Normalized DCG)"""
    actual_dcg = dcg(scores, k)
    ideal_dcg = dcg(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def l_cal(img1, img2):
    """计算 L2 和 L∞ 范数"""
    noise = (img1 - img2).view(-1)
    l2 = torch.norm(noise, p=2)
    l_inf = torch.norm(noise, p=float('inf'))
    return l2, l_inf

def set_seed(seed: int = 1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def clamp(x: torch.Tensor, low: float, high: float):
    return torch.max(torch.min(x, torch.tensor(high, device=x.device)), torch.tensor(low, device=x.device))


@torch.no_grad()
def hamming_distance_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Efficient batched Hamming distance between {-1,+1} tensors
    a: (B, D) , b: (N, D) -> (B,N)
    bit trick: (D - (a·bᵀ)) / 2
    """
    assert a.dim() == 2 and b.dim() == 2
    prod = torch.matmul(a, b.t())
    d = a.size(1)
    return 0.5 * (d - prod)