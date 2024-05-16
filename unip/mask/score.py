import torch
import torch.nn as nn


## score function
def rand_score(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.rand_like(module.weight.data).abs()
    return score_dict


def randn_score(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.randn_like(module.weight.data).abs()
    return score_dict


def l1_score(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = module.weight.data.to(torch.device("cpu")).abs()
    return score_dict


def snip_score(model, dataloader):
    device = next(model.parameters()).device
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.zeros_like(module.weight.data)
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        torch.nn.functional.cross_entropy(output, target).backward()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                score_dict[name] += (
                    module.weight.grad.data.detach().abs()
                    * module.weight.data.detach().abs()
                )
        model.zero_grad()
        break
    return score_dict


@torch.no_grad()
def linearize(model):
    signs_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            signs_dict[name] = module.weight.data.sign()
            module.weight.data.abs_()
    return signs_dict


@torch.no_grad()
def nonlinearize(model, signs_dict):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.mul_(signs_dict[name])


def synflow_score(model, example_data):
    device = next(model.parameters()).device
    model.eval()
    inputs = torch.ones_like(example_data).to(device)
    output = model(inputs)
    torch.sum(output).backward()
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = (
                module.weight.grad.data.detach().abs()
                * module.weight.data.detach().abs()
            )
    model.zero_grad()
    model.train()
    return score_dict
