import torch
import torch.nn as nn


def name2normfn(name):
    return globals()[name]


def znorm(score):
    return (score - score.mean()) / score.std()


def minmaxnorm(score):
    return (score - score.min()) / (score.max() - score.min())


def decimalnorm(score):
    return score / 10 ** int(torch.log10(score).item())


def meannorm(score):
    return (score - score.mean()) / (score.max() - score.min())


def unitnorm(score):
    return score / torch.norm(score, 1)


def robustnorm(score):
    return score / (torch.quantile(score, 0.75) - torch.quantile(score, 0.25))
