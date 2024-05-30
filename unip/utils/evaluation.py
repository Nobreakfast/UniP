import torch
from thop import profile, clever_format


def to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, list):
        return [to_device(v, device) for v in input]
    elif isinstance(input, tuple):
        return tuple(to_device(v, device) for v in input)
    elif isinstance(input, dict):
        return {k: to_device(v, device) for k, v in input.items()}
    else:
        return input


def cal_flops(model, example_input, device):
    model.to(device)
    if isinstance(example_input, torch.Tensor):
        example_input = [
            to_device(example_input, device),
        ]
    elif isinstance(example_input, dict):
        example_input = [to_device(v, device) for v in example_input.values()]
    elif isinstance(example_input, (list, tuple)):
        example_input = [to_device(v, device) for v in example_input]
    flops, params = profile(model, inputs=example_input, verbose=False)
    return flops, params, clever_format([flops, params], "%.3f")
