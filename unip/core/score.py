import torch
import torch.nn as nn

from unip.core.node import *


def rand(group):  # note: nodes is saved for other score functions
    return torch.rand(group.length)


def randn(group):
    return -torch.randn(group.length).abs()


def weight_sum_l1_out(group):
    score = torch.zeros(group.length)
    for n in group.nodes:
        if not n.is_prunable or isinstance(n, InInNode):
            continue
        for param in n.param:
            tuple_weight_index = tuple([i for i in range(param.dim()) if i != 0])
            score += param.abs().sum(dim=tuple_weight_index)
    return -score


def name2score(name):
    return globals()[name]
