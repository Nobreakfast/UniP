import torch
import torch.nn as nn

from unip.core.node import InOutNode, OutOutNode, CustomNode
from unip.utils.data_type import DIM_IN, DIM_OUT, IDX_IN, IDX_OUT


def save_model(model, path):
    assert isinstance(
        model, nn.Module
    ), f"model must be a nn.Module, but got {type(model)}"
    torch.save(model, path)


def load_model(path):
    model = torch.load(path)
    assert isinstance(
        model, nn.Module
    ), f"model must be a nn.Module, but got {type(model)}"
    return model


def save_model_dict(model, key2node, path):
    prune_dict = {}
    for key, node in key2node.items():
        if isinstance(node, (InOutNode, OutOutNode, CustomNode)):
            assert hasattr(
                node, "get_attr"
            ), f"{node} does not have get_attr() method, please use save_model() instead."
            prune_dict[key] = node.get_attr()
    torch.save(prune_dict, path)


def load_model_dict(model, path):
    prune_dict = torch.load(path)
    for key, attrs in prune_dict.items():
        module = _getattr(model, key)
        for attr, value in attrs.items():
            instance, attr = _getattr_prev(module, attr)
            setattr(instance, attr, value)
    return model


def _getattr(src, attr):
    attr_list = attr.split(".")
    if len(attr_list) == 1:
        return getattr(src, attr)
    src = getattr(src, attr_list[0])
    return _getattr(src, ".".join(attr_list[1:]))


def _getattr_prev(src, attr):
    attr_list = attr.split(".")
    if len(attr_list) == 1:
        return src, attr
    src = getattr(src, attr_list[0])
    return _getattr_prev(src, ".".join(attr_list[1:]))
