# --------------------------------------------------------
# BoundingBox Rectangle Utils
# Licensed under The MIT License
# Written by limengyao(mengyao.lmy@alibaba-inc.com)
# --------------------------------------------------------
#!/usr/bin/python

import torch
import numpy as np


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def torch_to_img(img):
    img = to_numpy(torch.squeeze(img, 0))
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img
