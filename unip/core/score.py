import torch
import torch.nn as nn
from unip.utils.data_type import DEVICE


def name2scorefn(name):
    return globals()[name]


def l1(params: list, length: int, dim: int = 0):
    score = torch.zeros(length).to(DEVICE)
    for param in params:
        if len(param.shape) > 1:
            norm_dim = [d for d in range(len(param.shape)) if d != dim]
            score += torch.norm(param, 1, dim=norm_dim)
        else:
            score += torch.abs(param)
    return score


def rand(params: list, length: int, dim: int = 0):
    return torch.rand(length).to(DEVICE)


def randn(params: list, length: int, dim: int = 0):
    return torch.randn(length).to(DEVICE)
