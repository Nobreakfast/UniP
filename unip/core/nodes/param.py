
"""
ParamNode class
"""

import torch
import torch.nn as nn
from .base import ParamNode

class DummyNode(ParamNode):
    def __init__(self, name: str, gradfn, param):
        super().__init__(name, None, gradfn)
        self.param = param
        self.shape = param.shape


class InputNode(DummyNode):
    def __init__(self, name: str, param):
        super().__init__(name, None, param)
        self.out_channels = self.shape[1]
        self.in_channels = self.shape[1]


class OutputNode(DummyNode):
    def __init__(self, name: str, gradfn, param):
        super().__init__(name, gradfn, param)
        self.out_channels = self.shape[1]
        self.in_channels = self.shape[1]


class BundleNode(DummyNode):
    def __init__(self, name: str, param):
        super().__init__(name, None, param)
        if len(self.shape) == 1:
            self.in_channels = self.shape[0]
            self.out_channels = self.shape[0]
        else:
            self.in_channels = self.shape[1]
            self.out_channels = self.shape[0]
        if self.out_channels != 1:
            self.prunable_param = param


