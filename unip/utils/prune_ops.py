import torch
import torch.nn as nn

from .data_type import DIM_IN, DIM_OUT, IDX_IN, IDX_OUT


def get_saved_idx(prune_idx, length):
    if prune_idx == []:
        return torch.arange(length)
    else:
        return torch.LongTensor([i for i in range(length) if i not in prune_idx])


def prune_param(param, saved_idx, dim):
    param.data = param.data.index_select(dim, saved_idx)
    if param.grad is not None:
        param.grad.data = param.grad.data.index_select(dim, saved_idx)


def prune_conv(conv, saved_idx, prune_dim):
    prune_param(conv.weight, saved_idx, prune_dim)
    if conv.bias is not None and prune_dim == DIM_OUT:
        prune_param(conv.bias, saved_idx, prune_dim)
    if prune_dim == DIM_IN:
        conv.in_channels = len(saved_idx)
    elif prune_dim == DIM_OUT:
        conv.out_channels = len(saved_idx)


def prune_transposeconv(conv, saved_idx, prune_dim):
    prune_param(conv.weight, saved_idx, 1 - prune_dim)
    if conv.bias is not None and prune_dim == DIM_OUT:
        prune_param(conv.bias, saved_idx, 0)
    if prune_dim == DIM_IN:
        conv.in_channels = len(saved_idx)
    elif prune_dim == DIM_OUT:
        conv.out_channels = len(saved_idx)


def prune_bundle(param, saved_idx, prune_dim):
    param.data = param.data.index_select(prune_dim, saved_idx)
    if param.grad is not None:
        param.grad.data = param.grad.data.index_select(prune_dim, saved_idx)


def prune_fc(fc, saved_idx, prune_dim):
    prune_param(fc.weight, saved_idx, prune_dim)
    if fc.bias is not None and prune_dim == DIM_OUT:
        prune_param(fc.bias, saved_idx, prune_dim)
    if prune_dim == DIM_IN:
        fc.in_features = len(saved_idx)
    elif prune_dim == DIM_OUT:
        fc.out_features = len(saved_idx)


def prune_emb(emb, saved_idx, prune_dim):
    prune_param(emb.weight, saved_idx, prune_dim)
    if prune_dim == DIM_IN:
        emb.num_embeddings = len(saved_idx)
    elif prune_dim == DIM_OUT:
        emb.embedding_dim = len(saved_idx)


def prune_batchnorm(norm, saved_idx, prune_dim):
    assert prune_dim == DIM_OUT
    prune_param(norm.weight, saved_idx, prune_dim)
    if norm.bias is not None:
        prune_param(norm.bias, saved_idx, prune_dim)
    norm.running_mean = norm.running_mean.data[saved_idx]
    norm.running_var = norm.running_var.data[saved_idx]
    norm.num_features = len(saved_idx)


def prune_layernorm(norm, saved_idx, prune_dim):
    assert prune_dim == DIM_OUT
    prune_param(norm.weight, saved_idx, prune_dim)
    if norm.bias is not None:
        prune_param(norm.bias, saved_idx, prune_dim)
    norm.normalized_shape = (len(saved_idx),)


def prune_groupnorm(norm, saved_idx, prune_dim):
    assert prune_dim == DIM_OUT
    prune_param(norm.weight, saved_idx, prune_dim)
    if norm.bias is not None:
        prune_param(norm.bias, saved_idx, prune_dim)
    norm.num_channels = len(saved_idx)
    norm.num_groups = len(saved_idx)


def prune_groupconv(conv, saved_idx, prune_dim):
    assert prune_dim == DIM_OUT
    prune_param(conv.weight, saved_idx, prune_dim)
    if conv.bias is not None:
        prune_param(conv.bias, saved_idx, prune_dim)
    conv.in_channels = len(saved_idx)
    conv.out_channels = len(saved_idx)
    conv.groups = len(saved_idx)
