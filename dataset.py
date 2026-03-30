# -*- coding: utf-8 -*-
# dataset.py

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class OxfordParisDataset(Dataset):
    """
    load Oxford5k / Paris6k: split='db'/'query'。
    """

    def __init__(self, root_img, root_label, transform=None, split='db'):
        super(OxfordParisDataset, self).__init__()
        self.root_img = root_img
        self.root_label = root_label
        self.transform = transform
        self.split = split

        if 'oxford5k' in self.root_label.lower():
            self.dataset_name = 'oxford5k'
        else:
            self.dataset_name = 'paris6k'

        self.landmark_dict = {}
        import glob
        label_files = glob.glob(os.path.join(root_label, "*.txt"))
        for lf in label_files:
            base = os.path.basename(lf)
            parts = base.split('_')
            labeltype = parts[-1].replace('.txt', '')  # good/ok/junk/query
            landmark_id = '_'.join(parts[:-1])

            if landmark_id not in self.landmark_dict:
                self.landmark_dict[landmark_id] = {'good': [], 'ok': [], 'junk': [], 'query': []}
            with open(lf, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    self.landmark_dict[landmark_id][labeltype].append(line)

        self.samples = []
        self.queryinfo = {}
        for lm_id, cat_dict in self.landmark_dict.items():
            good_lst = cat_dict.get('good', [])
            ok_lst = cat_dict.get('ok', [])
            junk_lst = cat_dict.get('junk', [])
            if self.split == 'db':
                all_db = good_lst + ok_lst + junk_lst
                for imgid in all_db:
                    self.samples.append((imgid, lm_id))
            else:
                qlist = cat_dict.get('query', [])
                for line in qlist:
                    parts = line.split()
                    imgid = parts[0]
                    if len(parts) == 5:
                        x1, y1, x2, y2 = map(float, parts[1:])
                        self.queryinfo[imgid] = (x1, y1, x2, y2)
                    self.samples.append((imgid, lm_id))

        self.samples = list({(x, y) for (x, y) in self.samples})
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgid, lm_id = self.samples[idx]

        if self.dataset_name == 'oxford5k':
            realpath = os.path.join(self.root_img, imgid + ".jpg")
        else:
            # paris6k 形如 'paris_defense_000034'
            parts = imgid.split('_')
            if len(parts) >= 2:
                subfolder = parts[1]
                realpath = os.path.join(self.root_img, subfolder, imgid + ".jpg")
            else:
                realpath = os.path.join(self.root_img, imgid + ".jpg")

        img = Image.open(realpath).convert('RGB')

        if imgid in self.queryinfo:
            x1, y1, x2, y2 = self.queryinfo[imgid]
            w, h = img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        return img, 0


def get_oxfordparis_dataloader(dataset_name='oxford5k', split='db', batch_size=16, shuffle=False):
    """
    read Oxford/Paris
    dataset_name: 'oxford5k' / 'paris6k'
    split: 'db' or 'query'
    """
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    if dataset_name == 'oxford5k':
        root_img = './dataset/oxford5k'
        root_label = './dataset/oxford5k_label'
    else:
        root_img = './dataset/paris6k'
        root_label = './dataset/paris6k_label'

    ds = OxfordParisDataset(root_img, root_label, transform=transform, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


def get_dataloader(dataset_name, split='train', batch_size=32, shuffle=True):
    dname = dataset_name.lower()
    if dname in ['oxford5k_db', 'oxford5k_query']:
        sp = 'db' if 'db' in dname else 'query'
        return get_oxfordparis_dataloader('oxford5k', sp, batch_size, shuffle)

    if dname in ['paris6k_db', 'paris6k_query']:
        sp = 'db' if 'db' in dname else 'query'
        return get_oxfordparis_dataloader('paris6k', sp, batch_size, shuffle)

    import torchvision.transforms as T
    transform = None
    if dname == 'mnist':
        from torchvision.datasets import MNIST
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor()
        ])
        ds = MNIST(root='./dataset/MNIST', train=(split == 'train'), download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    elif dname == 'cifar10':
        from torchvision.datasets import CIFAR10
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        ds = CIFAR10(root='./dataset/CIFAR10', train=(split == 'train'), download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    else:
        raise NotImplementedError(f"Unknown dataset => {dataset_name}")
