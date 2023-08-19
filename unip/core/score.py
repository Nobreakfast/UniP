import torch
import torch.nn as nn


def rand(length, num_toprune):
    return torch.randperm(length)[:num_toprune]
