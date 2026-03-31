# hashing/train_substitute_ensemble.py
# -*- coding: utf-8 -*-

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

from hashing.CSQ import build_csq_backbone


def build_substitute_backbone(model_name='alexnet', out_dim=64):
    if model_name == 'alexnet':
        net = torchvision.models.alexnet(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, out_dim)
    elif model_name == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, out_dim)
    elif model_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, out_dim)
    elif model_name == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, out_dim)
    elif backbone == 'ViT':
        net = torchvision.models.vit_l_32(pretrained=True)
        net.classifier = nn.Linear(1024, out_dim)
    elif backbone == 'Clip':
         net, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
        model.load_state_dict(torch.load('path/to/clip_weights.pt'))
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        net.classifier = nn.Linear(1024, out_dim)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return net


def load_victim_model(victim_ckpt, backbone='resnet50', bit=64, device='cuda'):
    victim_net = build_csq_backbone(backbone, bit)
    victim_net.load_state_dict(torch.load(victim_ckpt, map_location=device), strict=False)
    victim_net.eval().to(device)
    return victim_net


def blackbox_query_qair_style(dataset_name, victim_model, device, z=None, k=None):
    if z is None:
        z = config.default_z
    if k is None:
        k = config.default_k

    from dataset import get_dataloader

    if dataset_name in ['oxford5k_db', 'paris6k_db']:
        db_loader = get_dataloader(dataset_name, 'db', batch_size=32, shuffle=False)
    else:
        db_loader = get_dataloader(dataset_name, 'train', batch_size=32, shuffle=False)

    print("[BlackBox Query] Extracting database features...")
    db_samples = []
    db_features = []

    with torch.no_grad():
        for batch_idx, (imgs, lbs) in enumerate(db_loader):
            imgs = imgs.to(device)
            feats = victim_model(imgs).cpu()

            for i in range(len(imgs)):
                db_samples.append((imgs[i].cpu(), lbs[i] if isinstance(lbs, torch.Tensor) else lbs))
                db_features.append(feats[i])

    db_features = torch.stack(db_features)
    print(f"[BlackBox Query] Database size: {len(db_samples)}")

    query_results = []
    initial_count = min(10, len(db_samples))  
    initial_indices = random.sample(range(len(db_samples)), initial_count)

    img_save_dir = os.path.join("./substitute_data_images", dataset_name)
    os.makedirs(img_save_dir, exist_ok=True)

    current_queries = initial_indices
    all_queried = set()

    for round_i in range(z):
        print(f"[BlackBox Query] Round {round_i + 1}/{z}, queries: {len(current_queries)}")
        next_queries = []

        for q_idx in current_queries:
            if q_idx in all_queried:
                continue
            all_queried.add(q_idx)

            q_img, q_lb = db_samples[q_idx]
            q_feat = db_features[q_idx]

            distances = torch.norm(db_features - q_feat.unsqueeze(0), dim=1)

            _, topk_indices = torch.topk(distances, k=min(k + 1, len(db_samples)), largest=False)
            topk_indices = topk_indices[1:]  # 排除自己

            q_img_name = f"query_r{round_i}_q{len(query_results)}.png"
            q_img_path = os.path.join(img_save_dir, q_img_name)
            torchvision.utils.save_image(q_img, q_img_path)

            query_entry = {
                'query_img': q_img_path,
                'query_feat': q_feat.numpy(),
                'query_label': q_lb.item() if isinstance(q_lb, torch.Tensor) else q_lb,
                'topk_results': []
            }

            for rank, idx in enumerate(topk_indices[:k]):
                idx = idx.item()
                t_img, t_lb = db_samples[idx]
                t_feat = db_features[idx]

                t_img_name = f"top_r{round_i}_q{len(query_results)}_rank{rank}.png"
                t_img_path = os.path.join(img_save_dir, t_img_name)
                torchvision.utils.save_image(t_img, t_img_path)

                query_entry['topk_results'].append({
                    'img_path': t_img_path,
                    'feat': t_feat.numpy(),
                    'label': t_lb.item() if isinstance(t_lb, torch.Tensor) else t_lb,
                    'rank': rank,
                    'distance': distances[idx].item()
                })

                if idx not in all_queried and idx not in next_queries:
                    next_queries.append(idx)

            query_results.append(query_entry)
        current_queries = next_queries[:min(200, len(next_queries))] 

    print(f"[BlackBox Query] Total queries collected: {len(query_results)}")
    return query_results, img_save_dir


class SubstituteDataset(Dataset):
    def __init__(self, query_results, transform=None):
        super().__init__()
        self.samples = []
        self.rankings = []

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        for query in query_results:
            q_img_path = query['query_img']
            q_feat = torch.tensor(query['query_feat'], dtype=torch.float32)

            self.samples.append((q_img_path, q_feat))

            topk_sorted = sorted(query['topk_results'], key=lambda x: x['rank'])

            for i in range(len(topk_sorted)):
                for j in range(i + 1, min(i + 5, len(topk_sorted))): 
                    self.rankings.append({
                        'query': q_img_path,
                        'xi': topk_sorted[i]['img_path'],
                        'xj': topk_sorted[j]['img_path'],
                        'query_feat': q_feat,
                        'xi_feat': torch.tensor(topk_sorted[i]['feat'], dtype=torch.float32),
                        'xj_feat': torch.tensor(topk_sorted[j]['feat'], dtype=torch.float32)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_feat = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        ranking_info = None
        if len(self.rankings) > 0 and random.random() < 0.7:  
            ranking_idx = random.randint(0, len(self.rankings) - 1)
            ranking_info = self.rankings[ranking_idx]

        return img_tensor, target_feat, ranking_info


def substitute_collate_fn(batch):
    imgs = []
    target_feats = []
    ranking_infos = []

    for img, feat, ranking_info in batch:
        imgs.append(img)
        target_feats.append(feat)
        ranking_infos.append(ranking_info)

    imgs = torch.stack(imgs, dim=0)
    target_feats = torch.stack(target_feats, dim=0)

    return imgs, target_feats, ranking_infos


def ranking_loss_with_margin(sub_feat_i, sub_feat_j, sub_feat_q, gamma=0.05):
    d_xi = torch.norm(sub_feat_i - sub_feat_q, dim=1)
    d_xj = torch.norm(sub_feat_j - sub_feat_q, dim=1)

    loss = torch.relu(d_xi - d_xj + gamma)
    return loss.mean()


def train_single_substitute(model_name, dataset, device, out_dim=64, epochs=20, lr=1e-4):
    net = build_substitute_backbone(model_name, out_dim).to(device)

    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    num_workers = min(4, os.cpu_count())
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers,
                        pin_memory=True, collate_fn=substitute_collate_fn)

    best_loss = float('inf')
    patience = 10
    no_improve_count = 0

    ranking_cache = {}

    for ep in range(epochs):
        net.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_rank_loss = 0.0
        step_cnt = 0

        for imgs, target_feats, ranking_infos in loader:
            imgs = imgs.to(device)
            target_feats = target_feats.to(device)

            optimizer.zero_grad()

            out = net(imgs)
            mse_loss = mse_criterion(out, target_feats)

            rank_loss = torch.tensor(0.0).to(device)
            rank_count = 0

            ranking_imgs = []
            ranking_targets = []

            for ranking_info in ranking_infos:
                if ranking_info is None:
                    continue

                cache_key = (ranking_info['xi'], ranking_info['xj'], ranking_info['query'])
                if cache_key not in ranking_cache:
                    try:

                        xi_img = Image.open(ranking_info['xi']).convert('RGB')
                        xj_img = Image.open(ranking_info['xj']).convert('RGB')
                        q_img = Image.open(ranking_info['query']).convert('RGB')

                        transform = dataset.transform
                        xi_tensor = transform(xi_img)
                        xj_tensor = transform(xj_img)
                        q_tensor = transform(q_img)

                        ranking_cache[cache_key] = (xi_tensor, xj_tensor, q_tensor)
                    except Exception as e:
                        continue

                xi_tensor, xj_tensor, q_tensor = ranking_cache[cache_key]
                ranking_imgs.extend([xi_tensor, xj_tensor, q_tensor])
                ranking_targets.append((0, 1, 2))

            if len(ranking_imgs) > 0:

                ranking_imgs_batch = torch.stack(ranking_imgs).to(device)
                ranking_feats = net(ranking_imgs_batch)

                for i, (xi_idx, xj_idx, q_idx) in enumerate(ranking_targets):
                    base_idx = i * 3
                    xi_feat = ranking_feats[base_idx + xi_idx].unsqueeze(0)
                    xj_feat = ranking_feats[base_idx + xj_idx].unsqueeze(0)
                    q_feat = ranking_feats[base_idx + q_idx].unsqueeze(0)

                    rank_loss += ranking_loss_with_margin(xi_feat, xj_feat, q_feat, gamma=0.05)
                    rank_count += 1

            if rank_count > 0:
                rank_loss = rank_loss / rank_count
                total_loss = mse_loss + 0.3 * rank_loss 
            else:
                total_loss = mse_loss
                rank_loss = torch.tensor(0.0)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_rank_loss += rank_loss.item()
            step_cnt += 1

        avg_loss = epoch_loss / step_cnt
        avg_mse = epoch_mse_loss / step_cnt
        avg_rank = epoch_rank_loss / step_cnt

        print(f"[Substitute-{model_name}] epoch={ep + 1}/{epochs}, "
              f"total_loss={avg_loss:.4f}, mse={avg_mse:.4f}, rank={avg_rank:.4f}")

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience and ep > epochs // 2:
            print(f"[Substitute-{model_name}] Early stopping at epoch {ep + 1}")
            break

    ranking_cache.clear()

    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mnist')
    parser.add_argument('--victim_ckpt', default='./csq_models/csq_mnist_resnet50_64.pth')
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--out_path', default='./checkpoints')
    parser.add_argument('--z', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    args = parser.parse_args()

    device = config.device

    print("[Main] Loading victim model...")
    victim_model = load_victim_model(args.victim_ckpt, args.backbone, args.out_dim, device)

    query_results_file = f"./query_results_{args.dataset_name}.pt"
    if os.path.exists(query_results_file):
        print(f"[Main] Loading existing query results from {query_results_file}")
        query_results = torch.load(query_results_file)
        img_save_dir = os.path.join("./substitute_data_images", args.dataset_name)
    else:
        print("[Main] Performing black-box queries...")
        query_results, img_save_dir = blackbox_query_qair_style(
            args.dataset_name, victim_model, device, z=args.z, k=args.k
        )

        torch.save(query_results, query_results_file)
        print(f"[Main] Saved query results to {query_results_file}")

    print("[Main] Creating substitute dataset...")
    sub_dataset = SubstituteDataset(query_results)
    print(f"[Main] Substitute dataset size: {len(sub_dataset)}")
    print(f"[Main] Number of ranking pairs: {len(sub_dataset.rankings)}")

    os.makedirs(args.out_path, exist_ok=True)

    all_models = ["alexnet", "vgg16", "resnet50", "densenet121","ViT","Clip"]

    print(f"[Main] Training substitute models: {all_models}")
    for model_name in all_models:
        save_path = os.path.join(args.out_path, f"substitute_{args.dataset_name}_{model_name}.pth")

        if os.path.exists(save_path):
            print(f"[Main] Skip {model_name}, already exists: {save_path}")
            continue

        print(f"\n[Main] Training {model_name}...")
        net = train_single_substitute(model_name, sub_dataset, device, args.out_dim, args.epochs)

        torch.save(net.state_dict(), save_path)
        print(f"[Main] Saved {model_name} to {save_path}")

    print("\n[Main] All substitute models trained successfully!")


if __name__ == "__main__":
    main()
