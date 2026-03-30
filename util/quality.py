#util/quality.py
# -*- coding: utf-8 -*-
import torch
import numpy as np
import lpips
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from .utils import l_cal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_on_vgg = lpips.LPIPS(net='vgg').to(device)

def compare_images(img1_path, img2_path):
    """
    ȫ��Ա�����ͼƬ��
    - MSE
    - PSNR
    - SSIM
    - LPIPS
    - l2, l_inf
    """
    transform = T.ToTensor()

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    img1_t = transform(img1).unsqueeze(0).to(device)
    img2_t = transform(img2).unsqueeze(0).to(device)

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    mse_val = mean_squared_error(img1_np, img2_np)
    psnr_val = peak_signal_noise_ratio(img1_np, img2_np, data_range=255)
    ssim_val = structural_similarity(img1_np, img2_np, channel_axis=2)

    lp1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)
    lp2 = lpips.im2tensor(lpips.load_image(img2_path)).to(device)
    lpips_val = loss_on_vgg(lp1, lp2).item()

    l2, l_inf = l_cal(img1_t, img2_t)
    return {
        "MSE": mse_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "LPIPS": lpips_val,
        "L2": l2.item(),
        "L_inf": l_inf.item()
    }
