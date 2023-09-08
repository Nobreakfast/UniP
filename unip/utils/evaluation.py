import torch
import time
from thop import profile, clever_format
from tqdm import trange


def __inference_dict(model, example_input):
    model(**example_input)


def __inference_list(model, example_input):
    model(*example_input)


def get_data(model, example_input, device):
    model.to(device)
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input.to(device)]
        fn = __inference_list
    elif isinstance(example_input, dict):
        for k, v in example_input.items():
            example_input[k] = v.to(device)
        fn = __inference_dict
    elif isinstance(example_input, (list, tuple)):
        example_input = [v.to(device) for v in example_input]
        fn = __inference_list
    return fn, model, example_input


def cal_flops(model, example_input, device="cpu"):
    inference_fn, model, example_input = get_data(model, example_input, device)
    inference_fn(model, example_input)
    flops, params = profile(model, inputs=example_input, verbose=False)
    print(clever_format([flops, params], "%.3f"))
    return flops, params


def cal_fps(model, example_input, device, times=1000, warmup=0):
    inference_fn, model, example_input = get_data(model, example_input, device)
    for i in trange(warmup):
        inference_fn(model, example_input)
    start = time.time()
    for i in trange(times):
        inference_fn(model, example_input)
    end = time.time()
    return times / (end - start)
