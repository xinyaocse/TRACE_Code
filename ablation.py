# ablation.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from args import get_args_parser
from util.utils import load_image, clamp, guide_loss
import torchvision
import torchvision.transforms as transforms
from IAE_augmentation import (
    load_substitute_models,
    get_ensemble_feature,
    get_hash_features,
    compute_h_metric,
    precompute_gallery_features,
    compute_mse_similarity,
    compute_h_metric_optimized,
    get_binary_hash
)
from model.model import Model, init_model
from TRACE_attack import dwt_init, iwt_init, get_multi_target_dwt
from dataset import get_dataloader
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


def auto_select_query_image(dataset_name: str, out_path: str):
    """自动从数据集中选择一张查询图像"""
    dl = get_dataloader(dataset_name, split='test', batch_size=1, shuffle=True)
    for imgs, lbs in dl:
        pil_img = torchvision.transforms.ToPILImage()(imgs[0])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224))
        ])
        pil_img = transform(pil_img)
        pil_img.save(out_path)
        print(f"[Ablation] auto-selected query image => {out_path}")
        break


def ablation_no_iae(args, device):
    """消融实验：不使用IAE增强，直接使用原始目标图像"""
    print("\n[Ablation] Running without IAE augmentation...")

    # 加载查询图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    xq = load_image(args.query_img).to(device)

    # 直接使用原始目标图像
    target_files = [os.path.join(args.target_imgs_dir, f) for f in os.listdir(args.target_imgs_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    target_files.sort()

    if not target_files:
        raise FileNotFoundError("[Ablation] No target images found in target_imgs_dir.")

    # 使用多个目标图像
    xt_list = []
    for i in range(min(len(target_files), args.m)):
        xt_path = target_files[i]
        xt = load_image(xt_path).to(device)
        xt_list.append(xt)

    print(f"[Ablation] Using {len(xt_list)} original target images")

    # 加载替代模型
    subs = load_substitute_models(args.dataset, args.substitute_dir, device)

    # 计算目标特征的平均值
    avg_tgt_features = {}
    with torch.no_grad():
        for name, model in subs.items():
            target_feats = []
            for xt in xt_list:
                f = get_hash_features(model, xt, model_name=name, binary=False)
                f = torch.tanh(f)
                f = F.normalize(f, p=2, dim=1)
                target_feats.append(f)
            avg_tgt_features[name] = torch.stack(target_feats).mean(dim=0)

    # 普通对抗攻击（不使用IAE增强的目标）
    delta = torch.zeros_like(xq, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta], lr=config.iae_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)

    best_loss = float('inf')
    best_delta = None

    # 准备gallery数据加载器用于H度量
    if args.dataset in ['oxford5k', 'paris6k']:
        gallery_dataset = args.dataset.replace('_db', '') + '_db'
    else:
        gallery_dataset = args.dataset
    gallery_loader = get_dataloader(gallery_dataset, 'train', batch_size=32, shuffle=False)

    # 预计算gallery特征
    gallery_features = precompute_gallery_features(subs, gallery_loader, device,
                                                   use_hash=True, binary=config.use_binary_hash)

    for step in range(args.max_iter):
        optimizer.zero_grad()

        adv_x = clamp(xq + delta, 0, 1)

        # 计算损失
        total_loss = 0

        # 需要梯度的特征计算
        for name, model in subs.items():
            adv_feat = get_hash_features(model, adv_x, model_name=name, binary=False)
            adv_feat = torch.tanh(adv_feat)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)

            query_feat = get_hash_features(model, xq, model_name=name, binary=False).detach()
            query_feat = torch.tanh(query_feat)
            query_feat = F.normalize(query_feat, p=2, dim=1)

            target_feat = avg_tgt_features[name].detach()

            # Triplet loss
            pos_dist = F.pairwise_distance(adv_feat, target_feat)
            neg_dist = F.pairwise_distance(adv_feat, query_feat)

            margin = 2.0
            triplet_loss = F.relu(pos_dist - neg_dist + margin)

            total_loss = total_loss + triplet_loss.mean() / len(subs)

        # 添加H度量
        if step < 50 or step % 10 == 0:
            with torch.no_grad():
                h_score = compute_h_metric_optimized(subs, adv_x, xq, gallery_features,
                                                     k=10, use_hash=True, binary=config.use_binary_hash)
            total_loss = total_loss - args.lambda_j * h_score

        # 反向传播
        total_loss.backward()

        # 限制梯度
        delta.grad.data.clamp_(-config.eps, config.eps)
        optimizer.step()
        scheduler.step()

        # 限制扰动幅度
        delta.data = clamp(delta.data, -config.eps, config.eps)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_delta = delta.clone().detach()

        if step % 50 == 0:
            print(f"  Step {step}/{args.max_iter}, loss={total_loss.item():.4f}")

    # 生成最终对抗样本
    adv_final = clamp(xq + best_delta, 0, 1)

    # 反归一化
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    adv_final_denorm = denorm(adv_final[0]).unsqueeze(0)

    os.makedirs(args.outputpath, exist_ok=True)
    sp = os.path.join(args.outputpath, "adv_noIAE.png")
    torchvision.utils.save_image(adv_final_denorm, sp)
    print(f"[Ablation] No IAE result => {sp}, best_loss={best_loss:.4f}")


def ablation_no_rie(args, device):
    """消融实验：不使用RIE，使用简单的对抗攻击"""
    print("\n[Ablation] Running without RIE module...")

    # 加载查询图像
    xq = load_image(args.query_img).to(device)

    # 使用IAE增强后的目标图像
    xt_list = []
    iae_files = sorted([f for f in os.listdir(args.IAE_path) if f.startswith('IAE_')])

    if len(iae_files) == 0:
        raise FileNotFoundError(f"[Ablation] Cannot find IAE results in: {args.IAE_path}")

    for iae_file in iae_files[:args.m]:
        xt_path = os.path.join(args.IAE_path, iae_file)
        xt = load_image(xt_path).to(device)
        xt_list.append(xt)

    print(f"[Ablation] Using {len(xt_list)} IAE-augmented target images")

    # 加载替代模型
    subs = load_substitute_models(args.dataset, args.substitute_dir, device)

    # 简单的对抗攻击（不使用RIE）
    delta = torch.zeros_like(xq, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)

    best_loss = float('inf')
    best_delta = None

    # 准备gallery数据加载器用于H度量
    if args.dataset in ['oxford5k', 'paris6k']:
        gallery_dataset = args.dataset.replace('_db', '') + '_db'
    else:
        gallery_dataset = args.dataset
    gallery_loader = get_dataloader(gallery_dataset, 'train', batch_size=32, shuffle=False)

    # 预计算gallery特征
    gallery_features = precompute_gallery_features(subs, gallery_loader, device,
                                                   use_hash=True, binary=config.use_binary_hash)

    # 预计算目标特征
    target_features_cache = {}
    with torch.no_grad():
        for name, model in subs.items():
            target_feats = []
            for xt in xt_list:
                feat = get_hash_features(model, xt, model_name=name, binary=False)
                feat = torch.tanh(feat)
                feat = F.normalize(feat, p=2, dim=1)
                target_feats.append(feat)
            target_features_cache[name] = torch.cat(target_feats, dim=0)

    for step in range(args.max_iter):
        optimizer.zero_grad()

        adv_x = clamp(xq + delta, 0, 1)

        total_loss = 0

        for name, model in subs.items():
            adv_feat = get_hash_features(model, adv_x, model_name=name, binary=False)
            adv_feat = torch.tanh(adv_feat)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)

            query_feat = get_hash_features(model, xq, model_name=name, binary=False).detach()
            query_feat = torch.tanh(query_feat)
            query_feat = F.normalize(query_feat, p=2, dim=1)

            target_feats = target_features_cache[name]
            avg_target_feat = target_feats.mean(dim=0, keepdim=True)

            # Triplet loss
            pos_dist = F.pairwise_distance(adv_feat, avg_target_feat)
            neg_dist = F.pairwise_distance(adv_feat, query_feat)

            margin = 2.0
            triplet_loss = F.relu(pos_dist - neg_dist + margin)

            total_loss = total_loss + triplet_loss.mean() / len(subs)

        if step < 50 or step % 10 == 0:
            with torch.no_grad():
                h_score = compute_h_metric_optimized(subs, adv_x, xq, gallery_features,
                                                     k=10, use_hash=True, binary=config.use_binary_hash)
            total_loss = total_loss - args.lambda_j * h_score

        total_loss.backward()

        delta.grad.data.clamp_(-config.eps, config.eps)
        optimizer.step()
        scheduler.step()

        delta.data = clamp(delta.data, -config.eps, config.eps)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_delta = delta.clone().detach()

        if step % 50 == 0:
            print(f"  Step {step}/{args.max_iter}, loss={total_loss.item():.4f}")

    adv_final = clamp(xq + best_delta, 0, 1)

    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    adv_final_denorm = denorm(adv_final[0]).unsqueeze(0)

    os.makedirs(args.outputpath, exist_ok=True)
    sp = os.path.join(args.outputpath, "adv_noRIE.png")
    torchvision.utils.save_image(adv_final_denorm, sp)
    print(f"[Ablation] No RIE result => {sp}, best_loss={best_loss:.4f}")


def ablation_main():
    parser = get_args_parser()
    args = parser.parse_args()
    device = config.device

    set_seed(1234)

    if (not args.query_img) or (not os.path.isfile(args.query_img)):
        fallback = f"./auto_query_{args.dataset}_ablation.png"
        if os.path.exists(fallback):
            args.query_img = fallback
        else:
            auto_select_query_image(args.dataset, out_path=fallback)
            args.query_img = fallback

    if args.no_IAE:

        ablation_no_iae(args, device)
    elif args.no_RIE:

        ablation_no_rie(args, device)
    else:
        print("[Ablation] Please specify --no_IAE or --no_RIE for ablation study")


def main():
    ablation_main()


if __name__ == "__main__":
    main()
