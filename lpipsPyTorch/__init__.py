import torch

from .modules.lpips import LPIPS


lpips_cache = {}

def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    key = f"{net_type}-{version}"
    if key not in lpips_cache:
        lpips_cache[key] = LPIPS(net_type, version).to(device)
    criterion = lpips_cache[key].to(device)
    return criterion(x, y)
