import torch.nn as nn
import torch
from PIL import Image
import torch.nn.functional as Fn
import torchvision.transforms.functional as F
import pickle
import os
import numpy as np
from torchvision.transforms import transforms

from pooling import gempooling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()]) 
gem = gempooling.GeMPooling(2048, pool_size=3, init_norm=3.0).to(device)


def dcg(scores, k):
    """Compute the Discounted Cumulative Gain (DCG) at k.

    Args:
        scores (list or np.array): The relevance scores.
        k (int): The rank position to evaluate DCG at.

    Returns:
        float: The DCG value.
    """
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def ndcg(scores, ideal_scores, k):
    """Compute the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        scores (list or np.array): The relevance scores.
        ideal_scores (list or np.array): The ideal relevance scores.
        k (int): The rank position to evaluate NDCG at.

    Returns:
        float: The NDCG value.
    """
    actual_dcg = dcg(scores, k)
    ideal_dcg = dcg(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def load_image(imgpath):
    image = Image.open(imgpath)
    image = tf(image)
    image = image.unsqueeze(0)
    return image.cuda()


def gemfeature(tensor):
    gem_feature = gem(tensor)
    gem_feature = gem_feature.flatten()
    return gem_feature.cuda()


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def l1_loss(output, bicubic_image):
    loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1).to(device)
        std = self.std.reshape(1, 3, 1, 1).to(device)
        return (input - mean) / std


def cifar_name(number):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return str(label_names[number])


def normal_r(output_r):
    r_max = torch.max(output_r)
    r_min = torch.min(output_r)
    r_mean = r_max - r_min
    output_r = (output_r - r_min) / r_mean
    return output_r


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def imglist(path, mat):
    dirpath = []
    for parent, dirname, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(mat):
                dirpath.append(os.path.join(parent, filename))
    return dirpath


def testlist(path):
    dirpath = []
    for parent, dirname, filenames in os.walk(path):
        for filename in filenames:
            if filename.find("result") != -1:
                dirpath.append(os.path.join(parent, filename))

    return dirpath


def l_cal(img1, img2):
    noise = (img1 - img2).flatten(start_dim=0)
    l2 = torch.sum(torch.pow(torch.norm(noise, p=2, dim=0), 2))
    l_inf = torch.sum(torch.norm(noise, p=float('inf'), dim=0))
    return l2, l_inf
