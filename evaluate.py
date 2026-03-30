import os

import torch
import torchvision
from torch import nn
from torchvision import transforms
from args import get_args_parser
import numpy as np
import tqdm
import clip

args = get_args_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint = torch.load("./checkpoints/{}/{}/{}_trained_checkpoint.pth".format(args.dataset, args.mode, args.model))
state_dict = checkpoint['model_state_dict']

if args.mode == "binary":
    dimension = 2
else:
    if args.dataset == "MNIST" | args.dataset == "MNIST" | args.dataset == "CIFAR10":
        dimension = 10
    if args.dataset == "Oxford5k" | args.dataset == "Paris6k":
        dimension = 11


def load_data():
    if args.dataset == "MNIST":
        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
            ])
    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
    train_data = torchvision.datasets.ImageFolder('./{}/{}-dataset/train'.format(args.dataset, args.mode),
                                                  transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False, num_workers=0)
    return train_loader


def output(dataloader, dev=device):
    if args.model == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, dimension)
        model.load_state_dict(state_dict)
        model.eval()

    if args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, dimension)
        model.load_state_dict(state_dict)
        model.eval()

    if args.model == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(4096, dimension)
        model.load_state_dict(state_dict)
        model.eval()

    if args.model == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(1024, dimension)
        model.load_state_dict(state_dict)
        model.eval()

    if args.model == "clip_i2t":  # 图生文
        clip_model, preprocess = clip.load("ViT-B/32", device=dev, jit=False)
        clip_model.eval()
        class CLIPImageFeature(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
            def forward(self, x):
                with torch.no_grad():
                    feat = self.clip_model.encode_image(x)
                    feat = feat.float()
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat
        model = CLIPImageFeature(clip_model)

    if args.model == "clip_t2i":
        clip_model, preprocess = clip.load("ViT-B/32", device=dev, jit=False)
        clip_model.eval()
        class CLIPTextFeature(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
            def forward(self, text_tokens):
                with torch.no_grad():
                    feat = self.clip_model.encode_text(text_tokens)
                    feat = feat.float()
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat
        model = CLIPTextFeature(clip_model)

    net = model.to(dev)
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(dev), targets.to(dev)

            if args.model.startswith("clip_t2i"):
                text_tokens = clip.tokenize(inputs).to(dev)
                outputs = net(text_tokens)
            else:
                outputs = net(inputs)

            full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)

    return torch.round(full_batch_output), full_batch_label


def evaluate(trn_binary, trn_label, tst_binary, tst_label):
    classes = np.max(tst_label) + 1
    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label == i)[0])[:50]]
            tst_sample_label = np.array([i]).repeat(50)
            continue
        else:
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[
                np.random.RandomState(seed=i).permutation(np.where(tst_label == i)[0])[:50]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(50)])
    query_times = tst_sample_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    with tqdm(total=query_times, desc="Query") as pbar:
        for i in range(query_times):
            query_label = tst_sample_label[i]
            query_binary = tst_sample_binary[i, :]
            query_result = np.count_nonzero(query_binary != trn_binary, axis=1)
            sort_indices = np.argsort(query_result)
            buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
            P = np.cumsum(buffer_yes) / Ns
            precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0] - 1]
            AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
            sum_tp = sum_tp + np.cumsum(buffer_yes)
            pbar.set_postfix({'Average Precision': '{0:1.5f}'.format(AP[i])})
            pbar.update(1)
    pbar.close()
    t_mAP = np.mean(AP)
    print("t-mAP:", t_mAP)