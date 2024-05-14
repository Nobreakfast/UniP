import torch
import torch.nn as nn
import abc

"""
Node class and its subclasses
"""


class BaseNode(abc.ABC):
    def __init__(self, name: str, module, gradfn):
        self.name = name
        self.module = module
        self.gradfn = gradfn
        self.next, self.prev = [], []
        self.input_group = None
        self.output_group = None
        self.prunable_param = None
        self.in_channels = None
        self.out_channels = None

    def add_group(self, group, type):
        if type == "input":
            self.input_group = group
        elif type == "output":
            self.output_group = group

    def add_next(self, next_node):
        if next_node not in self.next:
            self.next.append(next_node)
            next_node.add_prev(self)

    def add_prev(self, prev_node):
        self.prev.append(prev_node)

    def print(self, count=0, forward=True):
        nodes = self.next if forward else self.prev
        print("++" * count + self.name)
        for node in nodes:
            node.print(count + 1, forward)


class ModuleNode(BaseNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.prune_fn = None

    def prune(self, idx, dim=0):
        if idx == None or self.prune_fn ==None:
            return
        self.prune_fn(self.module, idx, dim)


class ActionNode(BaseNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)

    def pass_idx(self, idx, group=None):
        self.output_group.prune(idx)

class ParamNode(BaseNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)

    def pass_idx(self, idx, group=None):
        pass

