# hashing/CSQ.py
# -*- coding: utf-8 -*-
import open_clip
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import config
from PIL import Image
import random
import numpy as np


def set_seed(seed=1234):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_csq_backbone(backbone='resnet50', bit=64):
    """
    构建CSQ哈希backbone，将最后一层改为 bit 维输出
    """
    if backbone == 'alexnet':
        net = torchvision.models.alexnet(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, bit)
    elif backbone == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, bit)
    elif backbone == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, bit)
    elif backbone == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, bit)
    elif backbone == 'ViT':
        net = torchvision.models.vit_l_32(pretrained=True)
        net.classifier = nn.Linear(1024, bit)
    elif backbone == 'Clip':
         net, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
        model.load_state_dict(torch.load('path/to/clip_weights.pt'))
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        net.classifier = nn.Linear(1024, bit)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    return net


class CSQLoss(nn.Module):
    """
    增强的CSQ损失函数：分类CE + 量化约束 + 中心损失 + Triplet-boundary
    """

    def __init__(self, bit, n_class, center_weight=0.1, quant_weight=0.1, triplet_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bit = bit
        self.n_class = n_class
        self.center_weight = center_weight
        self.quant_weight = quant_weight
        self.triplet_weight = triplet_weight

        # 添加类别中心，用于让同类样本的哈希码更紧凑
        self.register_buffer('centers', torch.randn(n_class, bit))

    def forward(self, u, y, update_centers=True):
        # 1. 分类损失
        loss_ce = self.ce(u, y)

        # 2. 量化损失 - 让哈希码接近 +1 或 -1
        loss_quant = ((u.abs() - 1.0) ** 2).mean()

        # 3. 中心损失 - 让同类样本的哈希码更接近
        loss_center = 0.0
        if self.n_class > 1:  # 只在多类别时使用中心损失
            batch_size = u.size(0)

            # 获取每个样本对应的中心
            batch_centers = self.centers.index_select(0, y)

            # 计算到中心的距离
            loss_center = ((u - batch_centers) ** 2).sum() / batch_size

            # 更新中心（使用动量更新）
            if update_centers and self.training:
                with torch.no_grad():
                    # 计算每个类别的新中心
                    for i in range(self.n_class):
                        mask = (y == i)
                        if mask.sum() > 0:
                            # 动量更新
                            new_center = u[mask].mean(0)
                            self.centers[i] = 0.9 * self.centers[i] + 0.1 * new_center

        # 4. Triplet-boundary损失
        loss_triplet = 0.0
        if self.n_class > 1:
            batch_size = u.size(0)
            device = u.device

            for i in range(batch_size):
                anchor = u[i]
                anchor_label = y[i]

                # 找到正样本和负样本
                pos_mask = (y == anchor_label) & (torch.arange(batch_size, device=device) != i)
                neg_mask = (y != anchor_label)

                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    # 计算到所有正负样本的距离
                    pos_dists = torch.norm(u[pos_mask] - anchor.unsqueeze(0), p=2, dim=1)
                    neg_dists = torch.norm(u[neg_mask] - anchor.unsqueeze(0), p=2, dim=1)

                    # 选择最难的正负样本对
                    hardest_pos_dist = pos_dists.max()
                    hardest_neg_dist = neg_dists.min()

                    # Triplet loss with margin γ = 0.05
                    loss_triplet += F.relu(hardest_pos_dist - hardest_neg_dist + 0.05)

            if batch_size > 0:
                loss_triplet = loss_triplet / batch_size

        return loss_ce + self.quant_weight * loss_quant + \
            self.center_weight * loss_center + self.triplet_weight * loss_triplet


def get_dataloader_for_csq(dataset, batch_size):
    from dataset import get_oxfordparis_dataloader
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST, CIFAR10

    num_workers = min(8, os.cpu_count())  # 动态设置workers数量

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ds = MNIST(root='./dataset/MNIST', train=True, download=True, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        n_class = 10
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ds = CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        n_class = 10
    elif dataset == 'oxford5k_db':
        dset = get_oxfordparis_dataloader('oxford5k', 'db', batch_size=1, shuffle=True)
        ds_ = dset.dataset
        loader = DataLoader(ds_, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        n_class = 1
    elif dataset == 'paris6k_db':
        dset = get_oxfordparis_dataloader('paris6k', 'db', batch_size=1, shuffle=True)
        ds_ = dset.dataset
        loader = DataLoader(ds_, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        n_class = 1
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return loader, n_class


def train_csq(dataset, backbone, bit, epochs, center_w, batch_size):
    device = config.device
    loader, n_class = get_dataloader_for_csq(dataset, batch_size)
    net = build_csq_backbone(backbone, bit).to(device)

    # 使用增强的损失函数（包含Triplet-boundary）
    criterion = CSQLoss(bit, n_class, center_weight=center_w, quant_weight=0.01, triplet_weight=0.1).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    patience = 5
    no_improve_count = 0

    for ep in range(epochs):
        net.train()
        epoch_loss = 0
        step_cnt = 0

        for imgs, lbs in loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            optimizer.zero_grad()
            out = net(imgs)
            loss = criterion(out, lbs)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            step_cnt += 1

        avg_loss = epoch_loss / step_cnt
        print(
            f"[CSQ][{dataset}-{backbone}] epoch={ep + 1}/{epochs}, loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            os.makedirs('./csq_models', exist_ok=True)
            save_path = f'./csq_models/csq_{dataset}_{backbone}_{bit}.pth'
            torch.save(net.state_dict(), save_path)
            print(f"[CSQ] => saved best model => {save_path}")
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= patience and ep > epochs // 2:
            print(f"[CSQ] Early stopping at epoch {ep + 1}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--bit', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--center_loss_w', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    # 设置随机种子
    set_seed(1234)

    train_csq(args.dataset, args.backbone, args.bit, args.epochs, args.center_loss_w, args.batch_size)


if __name__ == "__main__":
    main()