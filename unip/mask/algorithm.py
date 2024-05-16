import torch
import torch.nn as nn

from unip.mask.unstructural import *
from unip.mask.score import *

DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


def name2pai(name):
    return globals()[name]


def rand(model, example_data, ratio, device):
    score_dict = rand_score(model)
    threshold = cal_threshold(score_dict, ratio)
    apply_prune(model, score_dict, threshold)


def randn(model, example_data, ratio, device):
    score_dict = randn_score(model)
    threshold = cal_threshold(score_dict, ratio)
    apply_prune(model, score_dict, threshold)


# def snip():
#     device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     score_dict = snip(model, trainloader)
#     threshold = cal_threshold(score_dict, args.prune)
#     apply_prune(model, score_dict, threshold)
#     model = model.to(torch.device("cpu"))


def synflow(model, example_data, ratio, device=DEVICE):
    sign_dict = linearize(model)
    iterations = 100
    model.to(device)
    for i in range(iterations):
        prune_ratio = ratio / iterations * (i + 1)
        score_dict = synflow_score(model, example_data)
        threshold = cal_threshold(score_dict, prune_ratio)
        if i != iterations - 1:
            apply_prune(model, score_dict, threshold)
            remove_mask(model)
        else:
            nonlinearize(model, sign_dict)
            apply_prune(model, score_dict, threshold)
    model.to(torch.device("cpu"))


def resynflow(model, example_data, ratio, device=DEVICE):
    sign_dict = linearize(model)
    iterations = 10
    model.to(device)
    for i in range(iterations):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                sn = torch.linalg.norm(
                    module.weight.view(module.weight.shape[0], -1), ord=2
                ).item()
                module.weight.data /= sn
            elif isinstance(module, nn.Linear):
                sn = torch.linalg.norm(module.weight, ord=2).item()
                module.weight.data /= sn
        prune_ratio = ratio / iterations * (i + 1)
        score_dict = synflow_score(model, example_data)
        threshold = cal_threshold(score_dict, prune_ratio)
        if i != iterations - 1:
            apply_prune(model, score_dict, threshold)
            remove_mask(model)
        else:
            nonlinearize(model, sign_dict)
            apply_prune(model, score_dict, threshold)
    model.to(torch.device("cpu"))
