# util/visualize.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torchvision


def tsne_visualize(features, labels, out_path="tsne_plot.png"):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    emb = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.savefig(out_path)
    plt.close()


def tsne_visualize_enhanced(features, labels, out_path, title="t-SNE Visualization"):
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels]

    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=min(30, len(features) - 1))
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(emb[mask, 0], emb[mask, 1],
                    c=[colors[idx]], label=label, s=100, alpha=0.7, edgecolors='black')

    plt.legend(loc='best', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def tsne_visualize_attack_process(features, labels, images, out_path, title="Attack Process"):
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=min(30, len(features) - 1))
    emb = tsne.fit_transform(features)

    fig = plt.figure(figsize=(16, 10))

    ax_main = plt.subplot(2, 1, 1)

    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[idx] for idx, label in enumerate(unique_labels)}

    for i, (x, y, label) in enumerate(zip(emb[:, 0], emb[:, 1], labels)):
        color = label_to_color[label]
        ax_main.scatter(x, y, c=[color], s=200, alpha=0.8, edgecolors='black', linewidth=2)
        ax_main.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

    if 'Query' in labels and 'Adversarial' in labels:
        query_idx = labels.index('Query')
        adv_idx = labels.index('Adversarial')
        ax_main.arrow(emb[query_idx, 0], emb[query_idx, 1],
                      emb[adv_idx, 0] - emb[query_idx, 0],
                      emb[adv_idx, 1] - emb[query_idx, 1],
                      head_width=0.5, head_length=0.3, fc='red', ec='red',
                      linewidth=2, alpha=0.5)

    ax_main.set_title(title, fontsize=16)
    ax_main.set_xlabel('t-SNE Component 1', fontsize=12)
    ax_main.set_ylabel('t-SNE Component 2', fontsize=12)
    ax_main.grid(True, alpha=0.3)

    ax_images = plt.subplot(2, 1, 2)
    ax_images.axis('off')

    n_images = len(images)
    for i, (img, label) in enumerate(zip(images, labels)):
        ax_img = plt.subplot(2, n_images, n_images + i + 1)

        denorm = torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        img_denorm = denorm(img)
        img_denorm = torch.clamp(img_denorm, 0, 1)

        img_np = img_denorm.permute(1, 2, 0).numpy()
        ax_img.imshow(img_np)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_retrieval_results(query_img, retrieved_imgs, retrieved_labels, save_path):
    fig = plt.figure(figsize=(15, 8))

    ax_query = plt.subplot(2, 6, 1)
    denorm = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    query_denorm = denorm(query_img[0])
    query_denorm = torch.clamp(query_denorm, 0, 1)
    ax_query.imshow(query_denorm.permute(1, 2, 0).cpu().numpy())
    ax_query.set_title('Query', fontsize=12, fontweight='bold')
    ax_query.axis('off')

    for i in range(min(10, len(retrieved_imgs))):
        ax = plt.subplot(2, 6, i + 3)
        img = retrieved_imgs[i]
        if len(img.shape) == 4:
            img = img[0]
        img_denorm = denorm(img)
        img_denorm = torch.clamp(img_denorm, 0, 1)
        ax.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f'#{i + 1}\nLabel: {retrieved_labels[i]}', fontsize=10)
        ax.axis('off')

    plt.suptitle('Image Retrieval Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_defense_comparison(results_dict, save_path):
    methods = list(results_dict.keys())
    defenses = list(results_dict[methods[0]].keys())

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, defense in enumerate(defenses):
        values = [results_dict[method][defense] for method in methods]
        offset = width * (i - len(defenses) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=defense)

    ax.set_xlabel('Attack Methods')
    ax.set_ylabel('Defense Rate (%)')
    ax.set_title('Defense Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
