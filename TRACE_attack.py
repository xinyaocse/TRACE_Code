#TRACE_attack.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from args import get_args_parser
from util.utils import clamp, load_image, guide_loss
from model.model import Model, init_model
from IAE_augmentation import (
    load_substitute_models,
    get_ensemble_feature,
    compute_h_metric_optimized,
    precompute_gallery_features,
    get_penultimate_features,
    get_hash_features,
    compute_mse_similarity,
    FEATURE_CACHE,
    GALLERY_FEATURE_CACHE,
    get_binary_hash,
    compute_hamming_distance
)
import torchvision
import torchvision.transforms as transforms
from dataset import get_dataloader
from PIL import Image
import numpy as np
import gc
import random


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_select_query_image(dataset_name: str, out_path: str = "./auto_query_TRACE.png"):

    print(f"[TRACE] Auto-selecting query image from {dataset_name}...")

    if dataset_name in ['oxford5k', 'paris6k']:
        split = 'query'
        dataset_name = dataset_name + '_query'
    else:
        split = 'test'

    try:
        dl = get_dataloader(dataset_name, split, batch_size=1, shuffle=True)
        for imgs, lbs in dl:
            pil_img = torchvision.transforms.ToPILImage()(imgs[0])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224))
            ])
            pil_img = transform(pil_img)
            pil_img.save(out_path)
            print(f"[TRACE] Auto-selected query image => '{out_path}'")
            return True
    except Exception as e:
        print(f"[TRACE] Error selecting query image: {e}")
        return False

    return False


def dwt_init(x):
    b, c, h, w = x.shape
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)


def iwt_init(x):
    b, c, h, w = x.shape
    out_c = c // 4

    x_LL = x[:, 0:out_c, :, :]
    x_HL = x[:, out_c:2 * out_c, :, :]
    x_LH = x[:, 2 * out_c:3 * out_c, :, :]
    x_HH = x[:, 3 * out_c:4 * out_c, :, :]

    out_h = h * 2
    out_w = w * 2
    h_ = torch.zeros([b, out_c, out_h, out_w], device=x.device)

    t1 = (x_LL - x_HL - x_LH + x_HH) / 4
    t2 = (x_LL - x_HL + x_LH - x_HH) / 4
    t3 = (x_LL + x_HL - x_LH - x_HH) / 4
    t4 = (x_LL + x_HL + x_LH + x_HH) / 4

    h_[:, :, 0::2, 0::2] = t1
    h_[:, :, 1::2, 0::2] = t2
    h_[:, :, 0::2, 1::2] = t3
    h_[:, :, 1::2, 1::2] = t4

    return h_


def get_multi_target_dwt(xt_list, num_targets=None):
    if num_targets is None:
        num_targets = config.num_targets_rie

    # 确保不超过可用目标数
    num_targets = min(num_targets, len(xt_list))

    indices = torch.randperm(len(xt_list))[:num_targets]
    dwt_targets = []

    for idx in indices:
        dwt_t = dwt_init(xt_list[idx])
        dwt_targets.append(dwt_t)

    dwt_concat = torch.cat(dwt_targets, dim=1)
    return dwt_concat, indices


def compute_TRACE_loss_optimized(adv_img, xq, xt_list, models_dict, gallery_features,
                                device, lambda_f=None, lambda_j=None, wj=None,
                                query_features_cache=None, use_hash=True, binary=False):
    if lambda_f is None:
        lambda_f = config.lamda_per
    if lambda_j is None:
        lambda_j = config.lamda_j_default
    if wj is None:
        wj = {name: 1.0 / len(models_dict) for name in models_dict}

    lf_loss = 0
    m = len(xt_list)
    if query_features_cache is None:
        query_features_cache = {}
        with torch.no_grad():
            for name, model in models_dict.items():
                if use_hash:
                    q_feat = get_hash_features(model, xq, model_name=name, binary=False)
                    q_feat = torch.tanh(q_feat)  # 使用tanh激活
                else:
                    q_feat = get_penultimate_features(model, xq, model_name=name)
                query_features_cache[name] = F.normalize(q_feat, p=2, dim=1).detach()
    target_features_cache = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            target_feats = []
            batch_size = 8
            for i in range(0, len(xt_list), batch_size):
                batch = torch.cat(xt_list[i:i + batch_size], dim=0)
                if use_hash:
                    feat = get_hash_features(model, batch, model_name=name, binary=False)
                    feat = torch.tanh(feat)
                else:
                    feat = get_penultimate_features(model, batch, model_name=name)
                feat = F.normalize(feat, p=2, dim=1)
                target_feats.append(feat)
            target_features_cache[name] = torch.cat(target_feats, dim=0)

    for name, model in models_dict.items():
        model_loss = 0
        if use_hash:
            adv_feat = get_hash_features(model, adv_img, model_name=name, binary=False)
            adv_feat = torch.tanh(adv_feat)
        else:
            adv_feat = get_penultimate_features(model, adv_img, model_name=name, require_grad=True)

        adv_feat = F.normalize(adv_feat, p=2, dim=1)
        query_feat = query_features_cache[name]
        target_feats = target_features_cache[name]
        avg_target_feat = target_feats.mean(dim=0, keepdim=True)

        # Triplet loss
        pos_dist = F.pairwise_distance(adv_feat, avg_target_feat)
        neg_dist = F.pairwise_distance(adv_feat, query_feat)

        margin = 2.0
        triplet_loss = F.relu(pos_dist - neg_dist + margin)

        model_loss = triplet_loss.mean()
        lf_loss += wj[name] * model_loss

    if gallery_features is not None:
        h_score = compute_h_metric_optimized(models_dict, adv_img, xq, gallery_features,
                                             k=10, use_hash=use_hash, binary=binary)
        lf_loss -= lambda_j * h_score

    return lambda_f * lf_loss


def compute_reconstruction_loss(adv_dwt, q_dwt):
    c = adv_dwt.shape[1] // 4

    alpha = {
        'll': config.lamda_low_frequency,
        'lh': 1.0,
        'hl': 1.0,
        'hh': 1.0
    }

    lr_loss = 0
    components = ['ll', 'lh', 'hl', 'hh']

    for i, comp in enumerate(components):
        comp_adv = adv_dwt[:, i * c:(i + 1) * c, :, :]
        comp_q = q_dwt[:, i * c:(i + 1) * c, :, :]
        lr_loss += alpha[comp] * torch.nn.functional.mse_loss(comp_adv, comp_q)

    return lr_loss


def TRACE_attack_main():
    parser = get_args_parser()
    args = parser.parse_args()
    device = config.device

    set_seed(1234)

    print("\n" + "=" * 60)
    print(f"TRACE Attack: {args.dataset} with {args.model}")
    print("=" * 60)

    INN_net = Model().to(device)
    init_model(INN_net)
    print("[TRACE] Initialized RIE network")

    query_tensor = None

    if args.query_img and os.path.isfile(args.query_img):
        print(f"[TRACE] Using provided query image: {args.query_img}")
        query_path = args.query_img
    else:
        iae_query_path = f"./auto_query_{args.dataset}_iae.png"
        auto_query_path = f"./auto_query_{args.dataset}_TRACE.png"

        if os.path.exists(iae_query_path):
            print(f"[TRACE] Using query image from IAE stage: {iae_query_path}")
            query_path = iae_query_path
        elif os.path.exists(auto_query_path):
            print(f"[TRACE] Using existing auto-selected query image: {auto_query_path}")
            query_path = auto_query_path
        else:
            print(f"[TRACE] No query image provided, auto-selecting...")
            if auto_select_query_image(args.dataset, auto_query_path):
                query_path = auto_query_path
            else:
                raise ValueError("[TRACE] Failed to select query image!")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    xq_pil = Image.open(query_path).convert('RGB')
    xq = transform(xq_pil).unsqueeze(0).to(device)
    print(f"[TRACE] Loaded query image: {query_path}")
    xt_list = []
    iae_files = sorted([f for f in os.listdir(args.IAE_path) if f.startswith('IAE_')])

    if len(iae_files) == 0:
        raise FileNotFoundError(f"[Error] No IAE results found in {args.IAE_path}")

    print(f"[TRACE] Loading {len(iae_files)} IAE-augmented target images...")
    for iae_file in iae_files[:args.m]: 
        iae_path = os.path.join(args.IAE_path, iae_file)
        xt_pil = Image.open(iae_path).convert('RGB')
        xt = transform(xt_pil).unsqueeze(0).to(device)
        xt_list.append(xt)

    print(f"[TRACE] Loaded {len(xt_list)} target images")
    dwt_q = dwt_init(xq)
    subs = load_substitute_models(args.dataset, args.substitute_dir, device)
    if args.dataset in ['oxford5k', 'paris6k']:
        gallery_dataset = args.dataset.replace('_db', '') + '_db'
    else:
        gallery_dataset = args.dataset
    gallery_loader = get_dataloader(gallery_dataset, 'train', batch_size=64, shuffle=False)
    gallery_features = precompute_gallery_features(subs, gallery_loader, device,
                                                   use_hash=True, binary=config.use_binary_hash)
    query_features_cache = {}
    with torch.no_grad():
        for name, model in subs.items():
            q_feat = get_hash_features(model, xq, model_name=name, binary=False)
            q_feat = torch.tanh(q_feat)
            query_features_cache[name] = F.normalize(q_feat, p=2, dim=1).detach()
    best_loss = float('inf')
    best_cat = None
    loss_history = []
    patience = 30
    no_improve_count = 0

    print(f"\n[TRACE] Starting optimization (max_iter={args.max_iter})...")

    dwt_targets, target_indices = get_multi_target_dwt(xt_list, num_targets=config.num_targets_rie)
    input_cat = torch.cat([dwt_q, dwt_targets], dim=1).detach().clone()
    input_cat.requires_grad = True

    optimizer = torch.optim.Adam([input_cat], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)

    for step in range(args.max_iter):
        if step > 0 and step % 10 == 0:
            dwt_targets, target_indices = get_multi_target_dwt(xt_list, num_targets=config.num_targets_rie)
            with torch.no_grad():
                c_q = dwt_q.shape[1]
                input_cat.data[:, c_q:] = dwt_targets.data

        out_dwt = INN_net(input_cat)
        c_in = dwt_q.shape[1]

        adv_dwt = out_dwt[:, :c_in, :, :]
        adv_img = iwt_init(adv_dwt)
        adv_img = clamp(adv_img, 0, 1)

        if step < 50 or step % 5 == 0: 
            f_loss = compute_TRACE_loss_optimized(
                adv_img, xq, xt_list, subs, gallery_features,
                device, lambda_f=config.lamda_per, lambda_j=args.lambda_j,
                query_features_cache=query_features_cache,
                use_hash=True, binary=config.use_binary_hash
            )
        else:
            f_loss = compute_TRACE_loss_optimized(
                adv_img, xq, xt_list, subs, None,
                device, lambda_f=config.lamda_per, lambda_j=args.lambda_j,
                query_features_cache=query_features_cache,
                use_hash=True, binary=False
            )
        lr_loss = compute_reconstruction_loss(adv_dwt, dwt_q)

        total_loss = f_loss + lr_loss

        optimizer.zero_grad()
        total_loss.backward()
        input_cat.grad.data.clamp_(-config.eps, config.eps)

        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_cat = input_cat.clone().detach()
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and step > 50:
            print(f"  Early stopping at step {step}")
            break

        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step}: total_loss={total_loss.item():.4f}, "
                  f"f_loss={f_loss.item():.4f}, lr_loss={lr_loss.item():.4f}, lr={current_lr:.6f}")
            if step % 100 == 0:
                FEATURE_CACHE.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_dwt = INN_net(best_cat)
    adv_dwt = out_dwt[:, :dwt_q.shape[1], :, :]
    adv_img = iwt_init(adv_dwt)
    adv_img = clamp(adv_img, 0, 1)
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    adv_img_denorm = denorm(adv_img[0]).unsqueeze(0)
    os.makedirs(args.outputpath, exist_ok=True)
    save_path = os.path.join(args.outputpath, "adv_final.png")
    torchvision.utils.save_image(adv_img_denorm, save_path)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.title('TRACE Attack Loss Curve')
    plt.grid(True)
    plt.savefig(os.path.join(args.outputpath, 'loss_curve.png'))
    plt.close()

    FEATURE_CACHE.clear()
    GALLERY_FEATURE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[TRACE] Attack completed!")
    print(f"  Output: {save_path}")
    print(f"  Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    TRACE_attack_main()

