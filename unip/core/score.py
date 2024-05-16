import torch
import torch.nn as nn
def name2scorefn(name):
    return globals()[name]


def l1(params: list, length:int, dim:int=0):
    score = torch.zeros(length)
    for param in params:
        norm_dim = [d for d in range(len(param.shape)) if d != dim]
        score += torch.norm(param, 1, dim=norm_dim)
    return score

def rand(params: list, length:int, dim:int=0):
    return torch.rand(length)

def randn(params: list, length:int, dim:int=0):
    return torch.randn(length)
