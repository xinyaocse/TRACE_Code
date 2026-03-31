# IAE_augmentation.py
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from args import get_args_parser
from util.utils import guide_loss, clamp
import torchvision
import torchvision.transforms as transforms
from dataset import get_dataloader
from PIL import Image
import numpy as np
from functools import lru_cache
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


# 全局特征缓存
class FeatureCache:

    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get(self, key):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]

        self.cache[key] = value
        self.access_count[key] = 1

    def clear(self):
        self.cache.clear()
        self.access_count.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


FEATURE_CACHE = FeatureCache(max_size=50)
GALLERY_FEATURE_CACHE = {}


def get_binary_hash(model, x):
    h = model(x)
    # 使用tanh激活后二值化
    h = torch.tanh(h)
    h_binary = torch.sign(h)
    return h_binary


def compute_hamming_distance(h1, h2):
    # h1: [batch1, dim], h2: [batch2, dim]
    batch1, dim = h1.shape
    batch2, _ = h2.shape

    h1_expand = h1.unsqueeze(1).expand(batch1, batch2, dim)
    h2_expand = h2.unsqueeze(0).expand(batch1, batch2, dim)

    hamming = 0.5 * (dim - (h1_expand * h2_expand).sum(dim=2))
    return hamming


def compute_mse_similarity(feat1, feat2):
    return torch.mean((feat1 - feat2) ** 2, dim=1)


def load_substitute_models(dataset_name, sub_dir, device):
    from hashing.train_substitute_ensemble import build_substitute_backbone
    models_dict = {}

    all_models = ["alexnet", "vgg16", "resnet50", "densenet121"]

    for name in all_models:
        ckpt_path = os.path.join(sub_dir, f"substitute_{dataset_name}_{name}.pth")

        if not os.path.exists(ckpt_path):
            victim_path = f"./csq_models/csq_{dataset_name}_{name}_64.pth"
            if os.path.exists(victim_path):
                print(f"[IAE] Using victim model as substitute for {name}")
                ckpt_path = victim_path
            else:
                print(f"[IAE] Warning: Missing checkpoint {ckpt_path}, skipping {name}")
                continue

        net = build_substitute_backbone(name, 64)
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            net.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[IAE] Error loading {name}: {e}")
            continue

        net.eval().to(device)

        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            test_output = net(test_input)
            if test_output.shape[1] != 64:
                print(f"[IAE] Error: {name} output dim is {test_output.shape[1]}, expected 64")
                continue

        models_dict[name] = net
        print(f"[IAE] Loaded substitute model: {name} (verified 64-dim output)")

    if len(models_dict) == 0:
        raise ValueError("[IAE] No valid substitute models found!")

    return models_dict


def get_hash_features(model, x, model_name=None, binary=False):
    h = model(x)
    if binary and config.use_binary_hash:
        h = torch.tanh(h)
        h = torch.sign(h)
    return h


def get_penultimate_features(model, x, model_name=None, require_grad=False):
    device = x.device
    batch_size = x.size(0)

    if require_grad:
        if hasattr(model, 'fc'):  # ResNet, DenseNet
            if model_name and 'densenet' in model_name.lower():
                features = model.features(x)
                out = F.relu(features)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                feat = torch.flatten(out, 1)
            else:
                modules = list(model.children())[:-1]
                temp_model = nn.Sequential(*modules)
                feat = temp_model(x)
                feat = feat.view(batch_size, -1)

        elif hasattr(model, 'classifier'):  # AlexNet, VGG
            features = model.features(x)
            if hasattr(model, 'avgpool'):
                features = model.avgpool(features)
            features = features.view(batch_size, -1)

            if isinstance(model.classifier, nn.Sequential):
                classifier_layers = list(model.classifier.children())[:-1]
                temp_classifier = nn.Sequential(*classifier_layers)
                feat = temp_classifier(features)
            else:
                feat = features
        else:
            modules = list(model.children())[:-1]
            temp_model = nn.Sequential(*modules)
            feat = temp_model(x)
            feat = feat.view(batch_size, -1)
    else:
        with torch.no_grad():
            if hasattr(model, 'fc'):  # ResNet, DenseNet
                if model_name and 'densenet' in model_name.lower():
                    features = model.features(x)
                    out = F.relu(features)
                    out = F.adaptive_avg_pool2d(out, (1, 1))
                    feat = torch.flatten(out, 1)
                else:
                    modules = list(model.children())[:-1]
                    temp_model = nn.Sequential(*modules)
                    feat = temp_model(x)
                    feat = feat.view(batch_size, -1)

            elif hasattr(model, 'classifier'):  # AlexNet, VGG
                features = model.features(x)
                if hasattr(model, 'avgpool'):
                    features = model.avgpool(features)
                features = features.view(batch_size, -1)

                if isinstance(model.classifier, nn.Sequential):
                    classifier_layers = list(model.classifier.children())[:-1]
                    temp_classifier = nn.Sequential(*classifier_layers)
                    feat = temp_classifier(features)
                else:
                    feat = features
            else:
                modules = list(model.children())[:-1]
                temp_model = nn.Sequential(*modules)
                feat = temp_model(x)
                feat = feat.view(batch_size, -1)

    return feat


def get_ensemble_feature(models_dict, x, device, use_hash=True, requires_grad=False, binary=False):
    x = x.to(device)
    features = []

    if requires_grad:
        for name, net in models_dict.items():
            if use_hash:
                f = get_hash_features(net, x, model_name=name, binary=binary)
            else:
                f = get_penultimate_features(net, x, model_name=name, require_grad=True)

            f = F.normalize(f, p=2, dim=1)
            features.append(f)
    else:
        with torch.no_grad():
            for name, net in models_dict.items():
                if use_hash:
                    f = get_hash_features(net, x, model_name=name, binary=binary)
                else:
                    f = get_penultimate_features(net, x, model_name=name, require_grad=False)

                f = F.normalize(f, p=2, dim=1)
                features.append(f)

    if len(features) == 0:
        raise ValueError("No valid features extracted")

    ensemble = torch.stack(features).mean(dim=0)
    return ensemble


def precompute_gallery_features(models_dict, gallery_loader, device, use_hash=True, binary=False):
    cache_key = f"{id(gallery_loader)}_{list(models_dict.keys())}_{use_hash}_{binary}"

    if cache_key in GALLERY_FEATURE_CACHE:
        return GALLERY_FEATURE_CACHE[cache_key]

    print(f"[IAE] Precomputing gallery features (use_hash={use_hash}, binary={binary})...")
    gallery_features = {}

    for name, model in models_dict.items():
        print(f"[IAE] Processing {name}...")
        all_feats = []

        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(gallery_loader):
                imgs = imgs.to(device)
                if use_hash:
                    feats = get_hash_features(model, imgs, model_name=name, binary=binary)
                else:
                    feats = get_penultimate_features(model, imgs, model_name=name, require_grad=False)

                all_feats.append(feats.cpu())

                if batch_idx % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if batch_idx % 100 == 0:
                    print(
                        f"    Processed {batch_idx * gallery_loader.batch_size}/{len(gallery_loader.dataset)} samples")

        gallery_features[name] = torch.cat(all_feats, dim=0)

        del all_feats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    GALLERY_FEATURE_CACHE[cache_key] = gallery_features
    print(f"[IAE] Gallery features computed and cached")

    return gallery_features


def compute_h_metric_optimized(models_dict, x1, x2, gallery_features, k=10, use_hash=True, binary=False):
    device = x1.device

    weights = torch.tensor(
        [(2 ** (k - i) - 1) / sum(2 ** (k - j) - 1 for j in range(1, k + 1))
         for i in range(1, k + 1)],
        dtype=torch.float32
    )

    h_scores = []

    with torch.no_grad():
        for name, model in models_dict.items():
            if use_hash:
                feat1 = get_hash_features(model, x1, model_name=name, binary=binary).cpu()
                feat2 = get_hash_features(model, x2, model_name=name, binary=binary).cpu()
            else:
                feat1 = get_penultimate_features(model, x1, model_name=name, require_grad=False).cpu()
                feat2 = get_penultimate_features(model, x2, model_name=name, require_grad=False).cpu()

            gallery_feats = gallery_features[name]

            if binary and config.use_binary_hash:
                dist1 = compute_hamming_distance(feat1, gallery_feats).squeeze(0)
                dist2 = compute_hamming_distance(feat2, gallery_feats).squeeze(0)

                _, topk1 = torch.topk(dist1, k=k, largest=False)
                _, topk2 = torch.topk(dist2, k=k, largest=False)
            else:
                feat1_norm = F.normalize(feat1, p=2, dim=1)
                feat2_norm = F.normalize(feat2, p=2, dim=1)
                gallery_norm = F.normalize(gallery_feats, p=2, dim=1)

                sim1 = torch.mm(feat1_norm, gallery_norm.t()).squeeze(0)
                sim2 = torch.mm(feat2_norm, gallery_norm.t()).squeeze(0)

                _, topk1 = torch.topk(sim1, k=k, largest=True)
                _, topk2 = torch.topk(sim2, k=k, largest=True)

            h_score = 0.0
            topk2_set = set(topk2.numpy())
            topk2_dict = {idx.item(): i for i, idx in enumerate(topk2)}

            for i in range(k):
                idx1 = topk1[i].item()
                if idx1 in topk2_set:
                    j = topk2_dict[idx1]
                    h_score += weights[i] * weights[j]

            h_scores.append(h_score.item() if isinstance(h_score, torch.Tensor) else h_score)

    return np.mean(h_scores)


def compute_average_feature_centroid(target_imgs, models_dict, device, use_hash=True, binary=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_features = {}

    print("[IAE] Loading target images for centroid computation...")
    all_imgs = []
    for img_path in target_imgs:
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(pil_img)
        all_imgs.append(img_tensor)

    batch_size = 8

    for name, model in models_dict.items():
        all_features = []

        with torch.no_grad():
            for i in range(0, len(all_imgs), batch_size):
                batch_imgs = all_imgs[i:i + batch_size]
                batch_tensor = torch.stack(batch_imgs).to(device)

                if use_hash:
                    feat = get_hash_features(model, batch_tensor, model_name=name, binary=binary)
                else:
                    feat = get_penultimate_features(model, batch_tensor, model_name=name, require_grad=False)

                feat = F.normalize(feat, p=2, dim=1)
                all_features.append(feat.cpu())

                del batch_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_features = torch.cat(all_features, dim=0)
        model_features[name] = all_features.mean(dim=0, keepdim=True)

        del all_features
        gc.collect()

    centroids = {name: feat.to(device) for name, feat in model_features.items()}

    return centroids


def compute_iae_augmentation(orig_img, centroids, query_img, models_dict,
                             gallery_features, device, steps=200, lr=None,
                             lambda_j=None, wj=None, eps=None, use_hash=True,
                             binary=False):
    if lr is None:
        lr = config.iae_lr
    if lambda_j is None:
        lambda_j = config.lamda_j_default
    if eps is None:
        eps = config.IAE_eps
    if wj is None:
        wj = {name: 1.0 / len(models_dict) for name in models_dict}

    orig_img = orig_img.to(device)
    query_img = query_img.to(device)

    delta = torch.zeros_like(orig_img, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_loss = float('inf')
    best_delta = None

    patience = 30
    no_improve_count = 0

    query_features = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            if use_hash:
                q_feat = get_hash_features(model, query_img, model_name=name, binary=False)  
                q_feat = torch.tanh(q_feat)  
            else:
                q_feat = get_penultimate_features(model, query_img, model_name=name)
            query_features[name] = F.normalize(q_feat, p=2, dim=1).detach()

    for step in range(steps):
        optimizer.zero_grad()

        adv_img = torch.clamp(orig_img + delta, 0, 1)

        total_loss = 0

        for name, model in models_dict.items():
            if use_hash:
                adv_feat = get_hash_features(model, adv_img, model_name=name, binary=False)
                adv_feat = torch.tanh(adv_feat) 
            else:
                adv_feat = get_penultimate_features(model, adv_img, model_name=name, require_grad=True)

            adv_feat = F.normalize(adv_feat, p=2, dim=1)
            query_feat = query_features[name]
            center_feat = centroids[name].detach()

            # 使用Triplet Loss with margin
            pos_dist = F.pairwise_distance(adv_feat, center_feat)
            neg_dist = F.pairwise_distance(adv_feat, query_feat)

            margin = 2.0 
            triplet_loss = F.relu(pos_dist - neg_dist + margin)

            model_loss = wj[name] * triplet_loss.mean()
            total_loss = total_loss + model_loss

        reg_loss = 0.01 * torch.norm(delta, p=2)
        total_loss = total_loss + reg_loss

        if step < 50 or step % 10 == 0:
            with torch.no_grad():
                h_score = compute_h_metric_optimized(models_dict, adv_img, query_img,
                                                     gallery_features, k=10, use_hash=use_hash,
                                                     binary=binary)
            h_loss = lambda_j * h_score
            total_loss = total_loss - h_loss

        total_loss.backward()

        delta.grad.data.clamp_(-eps, eps)

        optimizer.step()
        scheduler.step()

        delta.data = torch.clamp(delta.data, -eps, eps)

        current_loss = total_loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_delta = delta.clone().detach()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience and step > 50:
            print(f"  Early stopping at step {step}")
            break

        if step % 50 == 0:
            print(f"  Step {step}/{steps}, loss={current_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return best_delta, best_loss


def select_target_images_from_dataset(dataset_name, target_category, m, target_dir):
    dl = get_dataloader(dataset_name, 'train', batch_size=100, shuffle=True)
    os.makedirs(target_dir, exist_ok=True)

    original_images = []

    for imgs, lbs in dl:
        if dataset_name in ['mnist', 'cifar10']:
            mask = (lbs == target_category)
            for i in range(len(imgs)):
                if mask[i]:
                    original_images.append(imgs[i])
                    if len(original_images) >= m:
                        break
        else:  # oxford5k, paris6k
            for img in imgs:
                original_images.append(img)
                if len(original_images) >= m:
                    break

        if len(original_images) >= m:
            break

    print(f"[IAE] Found {len(original_images)} images of target category {target_category}")

    if len(original_images) < m:
        print(f"[IAE] Not enough target images ({len(original_images)}), using augmentation...")
        augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor()
        ])

        augmented = []
        for i in range(m - len(original_images)):
            idx = i % len(original_images)
            aug_img = augment_transform(original_images[idx])
            augmented.append(aug_img)
        original_images.extend(augmented)

    random.shuffle(original_images)

    saved_count = 0
    for i in range(min(m, len(original_images))):
        img = original_images[i]
        pil_img = transforms.ToPILImage()(img)
        pil_img = pil_img.resize((224, 224))
        pil_img.save(os.path.join(target_dir, f"target_{i}.png"))
        saved_count += 1

    print(f"[IAE] Saved {saved_count} target images to {target_dir}")
    return saved_count


def auto_select_query_image(dataset_name, output_path):
    print(f"[IAE] Auto-selecting query image from {dataset_name}...")

    if dataset_name in ['oxford5k', 'paris6k']:
        split = 'query'
        dataset_name = dataset_name + '_query'
    else:
        split = 'test'

    try:
        dl = get_dataloader(dataset_name, split, batch_size=1, shuffle=True)

        for imgs, lbs in dl:
            pil_img = transforms.ToPILImage()(imgs[0])
            pil_img = pil_img.resize((224, 224))
            pil_img.save(output_path)
            print(f"[IAE] Auto-selected query image saved to: {output_path}")
            return True
    except Exception as e:
        print(f"[IAE] Error selecting query image: {e}")
        return False

    return False


def load_image_from_path(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_img = Image.open(img_path).convert('RGB')
    return transform(pil_img).unsqueeze(0)


def IAE_target_augmentation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--target_category', type=int, default=7)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--substitute_dir', default='./checkpoints')
    parser.add_argument('--target_imgs_dir', default='./target_images_mnist')
    parser.add_argument('--IAE_path', default='./IAE_outputs_mnist')
    parser.add_argument('--query_img', default='', help='path of query image')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda_j', type=float, default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize IAE effect')
    parser.add_argument('--use_hash', action='store_true', default=True, help='Use hash features')
    parser.add_argument('--binary', action='store_true', default=False, help='Use binary hash')
    args = parser.parse_args()

    device = config.device

    set_seed(1234)

    if args.lr is None:
        args.lr = config.iae_lr
    if args.lambda_j is None:
        args.lambda_j = config.lamda_j_default

    if not os.path.isdir(args.target_imgs_dir) or len(os.listdir(args.target_imgs_dir)) < args.m:
        print(f"[IAE] Selecting {args.m} target images from dataset...")
        actual_count = select_target_images_from_dataset(args.dataset, args.target_category, args.m,
                                                         args.target_imgs_dir)
        if actual_count < args.m:
            print(f"[IAE] Warning: Only got {actual_count} target images, adjusting m to {actual_count}")
            args.m = actual_count

    target_imgs = []
    for f in sorted(os.listdir(args.target_imgs_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            target_imgs.append(os.path.join(args.target_imgs_dir, f))
    target_imgs = target_imgs[:args.m]

    if len(target_imgs) < args.m:
        print(f"[IAE] Warning: Only found {len(target_imgs)} target images, expected {args.m}")

    models_dict = load_substitute_models(args.dataset, args.substitute_dir, device)

    query_tensor = None

    if args.query_img and os.path.exists(args.query_img):
        print(f"[IAE] Using provided query image: {args.query_img}")
        query_tensor = load_image_from_path(args.query_img).to(device)
    else:
        auto_query_path = f"./auto_query_{args.dataset}_iae.png"
        if os.path.exists(auto_query_path):
            print(f"[IAE] Using existing auto-selected query image: {auto_query_path}")
            query_tensor = load_image_from_path(auto_query_path).to(device)
        else:
            print(f"[IAE] No query image provided, auto-selecting from dataset...")
            if auto_select_query_image(args.dataset, auto_query_path):
                query_tensor = load_image_from_path(auto_query_path).to(device)
            else:
                print(f"[IAE] Auto-selection failed, using first target image as query...")
                if target_imgs:
                    query_tensor = load_image_from_path(target_imgs[0]).to(device)
                    import shutil
                    shutil.copy(target_imgs[0], auto_query_path)
                else:
                    raise ValueError("[IAE] No target images available and cannot auto-select query image!")

    if query_tensor is None:
        raise ValueError("[IAE] Failed to load or select query image!")

    print("[IAE] Computing feature centroids of target images...")
    centroids = compute_average_feature_centroid(target_imgs, models_dict, device,
                                                 use_hash=args.use_hash, binary=args.binary)

    if args.dataset in ['oxford5k', 'paris6k']:
        gallery_dataset = args.dataset + '_db'
    else:
        gallery_dataset = args.dataset
    gallery_loader = get_dataloader(gallery_dataset, 'train', batch_size=32, shuffle=False)

    gallery_features = precompute_gallery_features(models_dict, gallery_loader, device,
                                                   use_hash=args.use_hash, binary=args.binary)

    os.makedirs(args.IAE_path, exist_ok=True)

    wj = {name: 1.0 / len(models_dict) for name in models_dict}

    augmented_imgs = []

    for i, target_path in enumerate(target_imgs):
        print(f"\n[IAE] Processing target image {i + 1}/{len(target_imgs)}: {target_path}")

        orig_tensor = load_image_from_path(target_path).to(device)

        best_delta, best_loss = compute_iae_augmentation(
            orig_tensor, centroids, query_tensor, models_dict,
            gallery_features, device, steps=args.steps, lr=args.lr,
            lambda_j=args.lambda_j, wj=wj, eps=config.IAE_eps,
            use_hash=args.use_hash, binary=args.binary
        )

        final_img = orig_tensor + best_delta
        final_img = torch.clamp(final_img, 0, 1)
        augmented_imgs.append(final_img)

        denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        final_img_denorm = denorm(final_img[0]).unsqueeze(0)

        outp = os.path.join(args.IAE_path, f"IAE_{i}.png")
        torchvision.utils.save_image(final_img_denorm, outp)
        print(f"[IAE] Saved => {outp}, best_loss={best_loss:.4f}")

        if i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    GALLERY_FEATURE_CACHE.clear()
    FEATURE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[IAE] Completed! Generated {len(augmented_imgs)} IAE-augmented images.")


def main():
    IAE_target_augmentation()


if __name__ == "__main__":
    main()
