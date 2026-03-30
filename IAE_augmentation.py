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
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 全局特征缓存
class FeatureCache:
    """统一的特征缓存管理器"""

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
        # 如果缓存满了，删除最少访问的项
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


# 全局缓存实例
FEATURE_CACHE = FeatureCache(max_size=50)
GALLERY_FEATURE_CACHE = {}


def get_binary_hash(model, x):
    """获取二值化哈希码"""
    h = model(x)
    # 使用tanh激活后二值化
    h = torch.tanh(h)
    h_binary = torch.sign(h)
    return h_binary


def compute_hamming_distance(h1, h2):
    """计算汉明距离"""
    # h1: [batch1, dim], h2: [batch2, dim]
    batch1, dim = h1.shape
    batch2, _ = h2.shape

    # 扩展维度以计算所有对的距离
    h1_expand = h1.unsqueeze(1).expand(batch1, batch2, dim)
    h2_expand = h2.unsqueeze(0).expand(batch1, batch2, dim)

    # 计算汉明距离
    hamming = 0.5 * (dim - (h1_expand * h2_expand).sum(dim=2))
    return hamming


def compute_mse_similarity(feat1, feat2):
    """计算MSE相似度"""
    return torch.mean((feat1 - feat2) ** 2, dim=1)


def load_substitute_models(dataset_name, sub_dir, device):
    """
    从 substitute_dir 中加载替代模型
    包括与受害模型相同的模型以提高迁移性
    """
    from hashing.train_substitute_ensemble import build_substitute_backbone
    models_dict = {}

    # 检查所有可用模型（包括受害模型）
    all_models = ["alexnet", "vgg16", "resnet50", "densenet121"]

    for name in all_models:
        ckpt_path = os.path.join(sub_dir, f"substitute_{dataset_name}_{name}.pth")

        # 如果文件不存在，尝试使用受害模型作为替代
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
            # 处理可能的键名不匹配
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            net.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[IAE] Error loading {name}: {e}")
            continue

        net.eval().to(device)

        # 验证输出维度
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
    """获取模型的哈希特征输出（最后一层）"""
    h = model(x)
    if binary and config.use_binary_hash:
        h = torch.tanh(h)
        h = torch.sign(h)
    return h


def get_penultimate_features(model, x, model_name=None, require_grad=False):
    """获取倒数第二层的特征"""
    device = x.device
    batch_size = x.size(0)

    if require_grad:
        # 需要梯度时的处理
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
        # 不需要梯度时使用no_grad
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
    """
    计算模型集合的平均特征
    requires_grad: 是否需要梯度
    binary: 是否返回二值化哈希
    """
    x = x.to(device)
    features = []

    if requires_grad:
        # 需要梯度时不使用no_grad
        for name, net in models_dict.items():
            if use_hash:
                f = get_hash_features(net, x, model_name=name, binary=binary)
            else:
                f = get_penultimate_features(net, x, model_name=name, require_grad=True)

            # 归一化特征
            f = F.normalize(f, p=2, dim=1)
            features.append(f)
    else:
        with torch.no_grad():
            for name, net in models_dict.items():
                if use_hash:
                    f = get_hash_features(net, x, model_name=name, binary=binary)
                else:
                    f = get_penultimate_features(net, x, model_name=name, require_grad=False)

                # 归一化特征
                f = F.normalize(f, p=2, dim=1)
                features.append(f)

    if len(features) == 0:
        raise ValueError("No valid features extracted")

    # 计算集成特征
    ensemble = torch.stack(features).mean(dim=0)
    return ensemble


def precompute_gallery_features(models_dict, gallery_loader, device, use_hash=True, binary=False):
    """预计算并缓存gallery特征"""
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

                # 立即转到CPU以释放GPU内存
                all_feats.append(feats.cpu())

                # 定期清理GPU内存
                if batch_idx % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 显示进度
                if batch_idx % 100 == 0:
                    print(
                        f"    Processed {batch_idx * gallery_loader.batch_size}/{len(gallery_loader.dataset)} samples")

        # 在CPU上合并特征
        gallery_features[name] = torch.cat(all_feats, dim=0)

        # 清理内存
        del all_feats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 缓存结果
    GALLERY_FEATURE_CACHE[cache_key] = gallery_features
    print(f"[IAE] Gallery features computed and cached")

    return gallery_features


def compute_h_metric_optimized(models_dict, x1, x2, gallery_features, k=10, use_hash=True, binary=False):
    """
    优化的H度量计算
    """
    device = x1.device

    # 预计算权重
    weights = torch.tensor(
        [(2 ** (k - i) - 1) / sum(2 ** (k - j) - 1 for j in range(1, k + 1))
         for i in range(1, k + 1)],
        dtype=torch.float32
    )

    h_scores = []

    with torch.no_grad():
        for name, model in models_dict.items():
            # 计算查询特征
            if use_hash:
                feat1 = get_hash_features(model, x1, model_name=name, binary=binary).cpu()
                feat2 = get_hash_features(model, x2, model_name=name, binary=binary).cpu()
            else:
                feat1 = get_penultimate_features(model, x1, model_name=name, require_grad=False).cpu()
                feat2 = get_penultimate_features(model, x2, model_name=name, require_grad=False).cpu()

            # 使用缓存的gallery特征
            gallery_feats = gallery_features[name]

            # 计算相似度
            if binary and config.use_binary_hash:
                # 使用汉明距离
                dist1 = compute_hamming_distance(feat1, gallery_feats).squeeze(0)
                dist2 = compute_hamming_distance(feat2, gallery_feats).squeeze(0)

                # 获取top-k（距离最小的）
                _, topk1 = torch.topk(dist1, k=k, largest=False)
                _, topk2 = torch.topk(dist2, k=k, largest=False)
            else:
                # 归一化特征
                feat1_norm = F.normalize(feat1, p=2, dim=1)
                feat2_norm = F.normalize(feat2, p=2, dim=1)
                gallery_norm = F.normalize(gallery_feats, p=2, dim=1)

                # 计算余弦相似度
                sim1 = torch.mm(feat1_norm, gallery_norm.t()).squeeze(0)
                sim2 = torch.mm(feat2_norm, gallery_norm.t()).squeeze(0)

                # 获取top-k（相似度最大的）
                _, topk1 = torch.topk(sim1, k=k, largest=True)
                _, topk2 = torch.topk(sim2, k=k, largest=True)

            # 计算H分数
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
    """计算目标样本的平均特征质心（每个模型独立计算）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 为每个模型计算独立的特征质心
    model_features = {}

    print("[IAE] Loading target images for centroid computation...")
    all_imgs = []
    for img_path in target_imgs:
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(pil_img)
        all_imgs.append(img_tensor)

    # 批量处理
    batch_size = 8

    for name, model in models_dict.items():
        all_features = []

        with torch.no_grad():
            for i in range(0, len(all_imgs), batch_size):
                batch_imgs = all_imgs[i:i + batch_size]
                batch_tensor = torch.stack(batch_imgs).to(device)

                # 使用哈希特征
                if use_hash:
                    feat = get_hash_features(model, batch_tensor, model_name=name, binary=binary)
                else:
                    feat = get_penultimate_features(model, batch_tensor, model_name=name, require_grad=False)

                # 归一化
                feat = F.normalize(feat, p=2, dim=1)
                all_features.append(feat.cpu())

                # 清理GPU内存
                del batch_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 在CPU上计算平均
        all_features = torch.cat(all_features, dim=0)
        model_features[name] = all_features.mean(dim=0, keepdim=True)

        # 清理内存
        del all_features
        gc.collect()

    # 每个模型使用自己的质心
    centroids = {name: feat.to(device) for name, feat in model_features.items()}

    return centroids


def compute_iae_augmentation(orig_img, centroids, query_img, models_dict,
                             gallery_features, device, steps=200, lr=None,
                             lambda_j=None, wj=None, eps=None, use_hash=True,
                             binary=False):
    """
    计算IAE增强 - 使用Triplet Loss避免负值
    """
    if lr is None:
        lr = config.iae_lr
    if lambda_j is None:
        lambda_j = config.lamda_j_default
    if eps is None:
        eps = config.IAE_eps
    if wj is None:
        wj = {name: 1.0 / len(models_dict) for name in models_dict}

    # 确保所有张量在GPU上
    orig_img = orig_img.to(device)
    query_img = query_img.to(device)

    delta = torch.zeros_like(orig_img, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_loss = float('inf')
    best_delta = None

    # 早停参数
    patience = 30
    no_improve_count = 0

    # 预计算查询特征
    query_features = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            if use_hash:
                q_feat = get_hash_features(model, query_img, model_name=name, binary=False)  # 训练时不二值化
                q_feat = torch.tanh(q_feat)  # 使用tanh激活
            else:
                q_feat = get_penultimate_features(model, query_img, model_name=name)
            query_features[name] = F.normalize(q_feat, p=2, dim=1).detach()

    for step in range(steps):
        optimizer.zero_grad()

        # 生成对抗样本
        adv_img = torch.clamp(orig_img + delta, 0, 1)

        total_loss = 0

        # 对每个替代模型计算损失
        for name, model in models_dict.items():
            # 计算特征
            if use_hash:
                adv_feat = get_hash_features(model, adv_img, model_name=name, binary=False)
                adv_feat = torch.tanh(adv_feat)  # 使用tanh激活
            else:
                adv_feat = get_penultimate_features(model, adv_img, model_name=name, require_grad=True)

            # 归一化
            adv_feat = F.normalize(adv_feat, p=2, dim=1)
            query_feat = query_features[name]
            center_feat = centroids[name].detach()

            # 使用Triplet Loss with margin
            pos_dist = F.pairwise_distance(adv_feat, center_feat)
            neg_dist = F.pairwise_distance(adv_feat, query_feat)

            margin = 2.0  # 较大的margin确保不会出现负损失
            triplet_loss = F.relu(pos_dist - neg_dist + margin)

            model_loss = wj[name] * triplet_loss.mean()
            total_loss = total_loss + model_loss

        # 添加正则化项
        reg_loss = 0.01 * torch.norm(delta, p=2)
        total_loss = total_loss + reg_loss

        # 计算H度量（降低频率）
        if step < 50 or step % 10 == 0:
            with torch.no_grad():
                h_score = compute_h_metric_optimized(models_dict, adv_img, query_img,
                                                     gallery_features, k=10, use_hash=use_hash,
                                                     binary=binary)
            # H度量作为惩罚项（值越大越好，所以用负号）
            h_loss = lambda_j * h_score
            total_loss = total_loss - h_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        delta.grad.data.clamp_(-eps, eps)

        # 更新参数
        optimizer.step()
        scheduler.step()

        # 限制扰动幅度
        delta.data = torch.clamp(delta.data, -eps, eps)

        # 记录最佳结果
        current_loss = total_loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_delta = delta.clone().detach()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 早停检查
        if no_improve_count >= patience and step > 50:
            print(f"  Early stopping at step {step}")
            break

        if step % 50 == 0:
            print(f"  Step {step}/{steps}, loss={current_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            # 定期清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return best_delta, best_loss


def select_target_images_from_dataset(dataset_name, target_category, m, target_dir):
    """从数据集中选择m个目标类别的图像"""
    dl = get_dataloader(dataset_name, 'train', batch_size=100, shuffle=True)
    os.makedirs(target_dir, exist_ok=True)

    original_images = []

    # 直接收集所有目标类别图像
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

    # 如果不够，使用数据增强
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

    # 随机打乱
    random.shuffle(original_images)

    # 保存图像
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
    """自动从数据集中选择一张查询图像"""
    print(f"[IAE] Auto-selecting query image from {dataset_name}...")

    # 根据数据集选择合适的split
    if dataset_name in ['oxford5k', 'paris6k']:
        split = 'query'
        dataset_name = dataset_name + '_query'
    else:
        split = 'test'

    try:
        dl = get_dataloader(dataset_name, split, batch_size=1, shuffle=True)

        for imgs, lbs in dl:
            # 保存第一张图像作为查询图像
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
    """从路径加载图像"""
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

    # 设置随机种子
    set_seed(1234)

    # 使用config中的默认值
    if args.lr is None:
        args.lr = config.iae_lr
    if args.lambda_j is None:
        args.lambda_j = config.lamda_j_default

    # 1. 准备目标图像
    if not os.path.isdir(args.target_imgs_dir) or len(os.listdir(args.target_imgs_dir)) < args.m:
        print(f"[IAE] Selecting {args.m} target images from dataset...")
        actual_count = select_target_images_from_dataset(args.dataset, args.target_category, args.m,
                                                         args.target_imgs_dir)
        if actual_count < args.m:
            print(f"[IAE] Warning: Only got {actual_count} target images, adjusting m to {actual_count}")
            args.m = actual_count

    # 2. 收集目标图像路径
    target_imgs = []
    for f in sorted(os.listdir(args.target_imgs_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            target_imgs.append(os.path.join(args.target_imgs_dir, f))
    target_imgs = target_imgs[:args.m]

    if len(target_imgs) < args.m:
        print(f"[IAE] Warning: Only found {len(target_imgs)} target images, expected {args.m}")

    # 3. 加载替代模型
    models_dict = load_substitute_models(args.dataset, args.substitute_dir, device)

    # 4. 处理查询图像
    query_tensor = None

    if args.query_img and os.path.exists(args.query_img):
        print(f"[IAE] Using provided query image: {args.query_img}")
        query_tensor = load_image_from_path(args.query_img).to(device)
    else:
        # 自动选择查询图像
        auto_query_path = f"./auto_query_{args.dataset}_iae.png"
        if os.path.exists(auto_query_path):
            print(f"[IAE] Using existing auto-selected query image: {auto_query_path}")
            query_tensor = load_image_from_path(auto_query_path).to(device)
        else:
            print(f"[IAE] No query image provided, auto-selecting from dataset...")
            if auto_select_query_image(args.dataset, auto_query_path):
                query_tensor = load_image_from_path(auto_query_path).to(device)
            else:
                # 如果自动选择失败，从目标图像中选择第一张作为查询图像
                print(f"[IAE] Auto-selection failed, using first target image as query...")
                if target_imgs:
                    query_tensor = load_image_from_path(target_imgs[0]).to(device)
                    # 保存这张图像作为查询图像
                    import shutil
                    shutil.copy(target_imgs[0], auto_query_path)
                else:
                    raise ValueError("[IAE] No target images available and cannot auto-select query image!")

    if query_tensor is None:
        raise ValueError("[IAE] Failed to load or select query image!")

    # 5. 计算目标图像的特征质心（每个模型独立）
    print("[IAE] Computing feature centroids of target images...")
    centroids = compute_average_feature_centroid(target_imgs, models_dict, device,
                                                 use_hash=args.use_hash, binary=args.binary)

    # 6. 准备gallery数据加载器并预计算特征
    if args.dataset in ['oxford5k', 'paris6k']:
        gallery_dataset = args.dataset + '_db'
    else:
        gallery_dataset = args.dataset
    gallery_loader = get_dataloader(gallery_dataset, 'train', batch_size=32, shuffle=False)

    # 预计算gallery特征
    gallery_features = precompute_gallery_features(models_dict, gallery_loader, device,
                                                   use_hash=args.use_hash, binary=args.binary)

    # 7. 对每个目标图像进行IAE增强
    os.makedirs(args.IAE_path, exist_ok=True)

    # 权重设置
    wj = {name: 1.0 / len(models_dict) for name in models_dict}

    augmented_imgs = []

    for i, target_path in enumerate(target_imgs):
        print(f"\n[IAE] Processing target image {i + 1}/{len(target_imgs)}: {target_path}")

        orig_tensor = load_image_from_path(target_path).to(device)

        # 计算IAE增强
        best_delta, best_loss = compute_iae_augmentation(
            orig_tensor, centroids, query_tensor, models_dict,
            gallery_features, device, steps=args.steps, lr=args.lr,
            lambda_j=args.lambda_j, wj=wj, eps=config.IAE_eps,
            use_hash=args.use_hash, binary=args.binary
        )

        # 生成最终的增强图像
        final_img = orig_tensor + best_delta
        final_img = torch.clamp(final_img, 0, 1)
        augmented_imgs.append(final_img)

        # 反归一化用于保存
        denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        final_img_denorm = denorm(final_img[0]).unsqueeze(0)

        # 保存增强后的图像
        outp = os.path.join(args.IAE_path, f"IAE_{i}.png")
        torchvision.utils.save_image(final_img_denorm, outp)
        print(f"[IAE] Saved => {outp}, best_loss={best_loss:.4f}")

        # 定期清理内存
        if i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 清理缓存
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