"""
ModuleNode class
"""
import torch
import torch.nn as nn

from .base import ModuleNode
from unip.utils.prune_ops import *

class InOutNode(ModuleNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


class NormNode(ModuleNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_channels = module.num_features
        self.out_channels = module.num_features
        self.prunable_param = module.weight
        self.prune_fn = prune_batchnorm


class ActivationNode(ModuleNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_channels = 1
        self.out_channels = 1

    def pass_idx(self, idx, group=None):
        # self.output_group.prune(idx)
        if len(self.output_group.nodes) == 1:
            self.output_group.prune(idx)
class InInNode(ModuleNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)

    def pass_idx(self, idx, group=None):
        pass



"""
ModuleNode::InOutNode class
"""


class ConvNode(InOutNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.prunable_param = module.weight
        self.prune_fn = prune_conv

class LinearNode(InOutNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_channels = module.in_features
        self.out_channels = module.out_features
        self.prunable_param = module.weight
        self.prune_fn = prune_fc


class LastLinearNode(InOutNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_channels = module.in_features
        self.out_channels = module.out_features
        self.prunable_param = module.weight
        self.prune_fn = prune_fc


class EmbeddingNode(InOutNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.num_embeddings = module.num_embeddings
        self.embedding_dim = module.embedding_dim
        self.in_channels = self.num_embeddings
        self.out_channels = self.embedding_dim
        self.prunable_param = module.weight
        self.prune_fn = prune_emb


"""
ModuleNode::InInNode class
"""


class AddNode(InInNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


class MatMulNode(InInNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


"""
ActionNode::DimChangeNode class
"""
