#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


window = None


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    global window
    if window is None:
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        #return ssim_map.mean(1).mean(1).mean(1)
        return ssim_map


def ssim_with_center_grad(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    global window
    if window is None:
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

    return _ssim_with_center_grad(img1, img2, window, window_size, channel, size_average)


def _ssim_with_center_grad(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    denom_l = mu1_sq + mu2_sq + C1
    l = (2 * mu1_mu2 + C1) / denom_l

    denom_cs = sigma1_sq + sigma2_sq + C2
    cs = (2 * sigma12 + C2) / denom_cs

    ssim_map = l * cs

    # get the center value of the Gaussian kernel
    g = window[:, 0, window_size // 2, window_size // 2]
    g = g.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    g = g.expand_as(mu1)

    # calculate the partial derivative w.r.t. the center pixel using the formulas of https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/backpropagation_math.pdf
    d_l_d_center = 2 * g * (mu2 - mu1 * l) / denom_l
    d_cs_d_center = (2 / denom_cs) * g * ((img2 - mu2) - cs * (img1 - mu1))
    center_grad_ssim_map = d_l_d_center * cs + d_cs_d_center * l

    if size_average:
        return ssim_map.mean(), center_grad_ssim_map.mean()
    else:
        #return ssim_map.mean(1).mean(1).mean(1)
        return ssim_map, center_grad_ssim_map


if __name__ == '__main__':
    C = 3
    H = 32
    W = 32

    x = torch.randn(1, C, H, W).cuda()
    y = torch.randn_like(x)

    x.requires_grad_(True)

    ssim_loss = ssim(x, y, size_average=False)
    ssim_loss_explicit, grad_x_explicit = ssim_with_center_grad(x, y, size_average=False)

    assert torch.allclose(ssim_loss, ssim_loss_explicit, atol=1e-6)

    for c in range(C):
        for h in range(H):
            for w in range(W):
                grad_x = torch.autograd.grad(
                    ssim_loss[0, c, h, w],
                    [x],
                    retain_graph=True
                )[0]

                assert torch.allclose(grad_x[0, c, h, w], grad_x_explicit[0, c, h, w], atol=1e-6)

    print("finished check")
