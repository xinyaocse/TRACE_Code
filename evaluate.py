# evaluate.py
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from args import get_args_parser
from util.utils import load_image
import config
from hashing.CSQ import build_csq_backbone
from util.quality import compare_images
from dataset import get_dataloader
from util.visualize import (tsne_visualize_enhanced, visualize_retrieval_results,
                            plot_defense_comparison)
import lpips
from IAE_augmentation import get_binary_hash, compute_hamming_distance


def extract_hash(model, x, binary=True):
    with torch.no_grad():
        f = model(x)
        if binary and config.use_binary_hash:
            f = torch.tanh(f)
            f = torch.sign(f)
    return f


def calculate_t_map(retrieved_indices, target_indices, total_db_size):
    relevant = np.zeros(total_db_size, dtype=bool)
    relevant[target_indices] = True

    ap = 0.0
    relevant_found = 0

    for i, idx in enumerate(retrieved_indices):
        if relevant[idx]:
            relevant_found += 1
            # Precision at i = relevant_found / (i + 1)
            precision_at_i = relevant_found / (i + 1)
            ap += precision_at_i

    total_relevant = len(target_indices)
    if total_relevant > 0:
        ap = ap / min(total_relevant, len(retrieved_indices))

    return ap


def calculate_prop_k(retrieved_indices, target_indices, k):
    top_k_indices = retrieved_indices[:k]
    target_set = set(target_indices)

    count = sum(1 for idx in top_k_indices if idx in target_set)
    return count / k


def evaluate_hash_retrieval(victim_model, adv_img_paths, dataset_name='mnist', k=10,
                            target_label=None, visualize=False, output_dir=None):
    device = config.device

    dataset_name = dataset_name.lower()
    if dataset_name in ['oxford5k', 'oxford5k_db']:
        db_dataset = 'oxford5k_db'
        db_split = 'db'
    elif dataset_name in ['paris6k', 'paris6k_db']:
        db_dataset = 'paris6k_db'
        db_split = 'db'
    elif dataset_name in ['mnist', 'cifar10']:
        db_dataset = dataset_name
        db_split = 'train'
    else:
        raise NotImplementedError(f"[Evaluate] Not supported dataset => {dataset_name}")

    print(f"[Evaluate] Loading database: {db_dataset} ({db_split})...")
    db_loader = get_dataloader(db_dataset, db_split, batch_size=32, shuffle=False)
    victim_model.eval().to(device)

    db_feats = []
    db_labels = []
    db_imgs = []

    with torch.no_grad():
        for imgs, lbs in db_loader:
            imgs = imgs.to(device)

            feats = extract_hash(victim_model, imgs, binary=config.use_binary_hash).cpu()
            db_feats.append(feats)

            if isinstance(lbs, torch.Tensor):
                db_labels.extend(lbs.tolist())
            else:
                db_labels.extend([lbs] * len(imgs))

            if visualize:
                db_imgs.extend([img.cpu() for img in imgs])

    db_feats = torch.cat(db_feats, dim=0)
    db_labels = np.array(db_labels)
    print(f"[Evaluate] Database size: {len(db_labels)}")

    if target_label is not None:
        target_indices = np.where(db_labels == target_label)[0]
        print(f"[Evaluate] Target class {target_label} has {len(target_indices)} samples")
    else:
        target_indices = []

    total_prop = 0.0
    total_map = 0.0
    n = 0

    for adv_path in adv_img_paths:
        if not os.path.exists(adv_path):
            print(f"[Evaluate] Missing file: {adv_path}")
            continue
        adv_img = load_image(adv_path).to(device)
        adv_hash = extract_hash(victim_model, adv_img, binary=config.use_binary_hash).cpu()

        if config.use_binary_hash:
            distances = compute_hamming_distance(adv_hash, db_feats).squeeze(0)
            sorted_indices = distances.argsort().numpy()
        else:
            distances = torch.norm(db_feats - adv_hash, dim=1, p=2)
            sorted_indices = distances.argsort().numpy()

        prop_k = calculate_prop_k(sorted_indices, target_indices, k)

        t_map = calculate_t_map(sorted_indices, target_indices, len(db_labels))

        total_prop += prop_k
        total_map += t_map
        n += 1

        print(f"  {os.path.basename(adv_path)}: prop@{k}={prop_k:.3f}, t-MAP={t_map:.3f}")

        if visualize and output_dir:
            retrieved_imgs = [db_imgs[idx] for idx in sorted_indices[:10]]
            retrieved_labels = [db_labels[idx] for idx in sorted_indices[:10]]
            vis_path = os.path.join(output_dir, f"retrieval_{os.path.basename(adv_path)}")
            visualize_retrieval_results(adv_img, retrieved_imgs, retrieved_labels, vis_path)

    if n > 0:
        return total_prop / n, total_map / n
    else:
        return None, None


def evaluate_victim_baseline(victim_model, dataset_name, target_label, k=10):
    device = config.device

    if dataset_name in ['mnist', 'cifar10']:
        test_loader = get_dataloader(dataset_name, 'test', batch_size=100, shuffle=False)
        train_loader = get_dataloader(dataset_name, 'train', batch_size=100, shuffle=False)
    else:
        return None, None
    print("[Evaluate] Computing victim model baseline mAP...")

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = extract_hash(victim_model, imgs, binary=True).cpu()
            all_feats.append(feats)
            all_labels.extend(labels.tolist())

        db_feats = torch.cat(all_feats)
        db_labels = np.array(all_labels)

        total_ap = 0
        count = 0

        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            query_feats = extract_hash(victim_model, imgs, binary=True).cpu()

            for i in range(len(imgs)):
                q_feat = query_feats[i:i + 1]
                q_label = labels[i].item()

                distances = compute_hamming_distance(q_feat, db_feats).squeeze(0)
                sorted_indices = distances.argsort().numpy()
                relevant = (db_labels == q_label)
                ap = 0.0
                relevant_found = 0

                for j, idx in enumerate(sorted_indices[:1000]):  # top-1000
                    if relevant[idx]:
                        relevant_found += 1
                        ap += relevant_found / (j + 1)

                if relevant.sum() > 0:
                    ap = ap / min(relevant.sum(), 1000)

                total_ap += ap
                count += 1

    mAP = total_ap / count
    print(f"[Evaluate] Victim model mAP = {mAP:.3f}")

    return mAP, None


def evaluate_image_quality(query_img_path, adv_img_paths):
    if not os.path.exists(query_img_path):
        print(f"[Evaluate] Query image not found: {query_img_path}")
        return None

    all_metrics = []

    for adv_path in adv_img_paths:
        if not os.path.exists(adv_path):
            continue

        metrics = compare_images(query_img_path, adv_path)
        all_metrics.append(metrics)

        print(f"  {os.path.basename(adv_path)}:")
        print(f"    L2={metrics['L2']:.3f}, L∞={metrics['L_inf']:.3f}")
        print(f"    SSIM={metrics['SSIM']:.3f}, LPIPS={metrics['LPIPS']:.3f}, PSNR={metrics['PSNR']:.2f}dB")

    return all_metrics


def evaluate_main():
    parser = get_args_parser()
    parser.add_argument('--adv_path', default='')
    parser.add_argument('--target_label', type=int, default=None)
    parser.add_argument('--eval_baseline', action='store_true', help='Evaluate victim baseline')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    device = config.device
    if not args.adv_path:
        args.adv_path = f"./TRACE_outputs_{args.dataset}_{args.model}/adv_final.png"
    victim_ckpt = f"./csq_models/csq_{args.dataset}_{args.model}_64.pth"
    if not os.path.exists(victim_ckpt):
        print(f"[Evaluate] Victim model not found: {victim_ckpt}")
        return

    net = build_csq_backbone(args.model, bit=64)
    net.load_state_dict(torch.load(victim_ckpt, map_location=device), strict=False)
    net.eval().to(device)
    output_dir = f"./evaluation_results_{args.dataset}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    if args.eval_baseline:
        print("\n[Evaluate] Evaluating victim model baseline...")
        mAP, _ = evaluate_victim_baseline(net, args.dataset, args.target_label, args.k)
        if mAP is not None and mAP < 0.9:
            print(f"[Warning] Victim model mAP ({mAP:.3f}) is below expected threshold (0.9)")
    print("\n[Evaluate] Retrieval Performance:")
    adv_paths = [args.adv_path]
    prop_k, t_map = evaluate_hash_retrieval(net, adv_paths, args.dataset, args.k,
                                            args.target_label, visualize=args.visualize,
                                            output_dir=output_dir)

    if prop_k is not None:
        print(f"\nFinal Results:")
        print(f"  prop@{args.k} = {prop_k:.3f}")
        print(f"  t-MAP = {t_map:.3f}")
    print("\n[Evaluate] Image Quality:")
    query_img = args.query_img or f"./auto_query_{args.dataset}.png"
    if os.path.exists(query_img):
        evaluate_image_quality(query_img, adv_paths)


if __name__ == "__main__":
    evaluate_main()
