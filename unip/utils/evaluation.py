import torch
from thop import profile, clever_format


def cal_flops(model, example_input, device):
    model.to(device)
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input.to(device)]
    elif isinstance(example_input, dict):
        example_input = [v.to(device) for v in example_input.values()]
    elif isinstance(example_input, (list, tuple)):
        example_input = [v.to(device) for v in example_input]
    flops, params = profile(model, inputs=example_input, verbose=False)
    print(clever_format([flops, params], "%.3f"))
    return flops, params
