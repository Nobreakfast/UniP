"""
ActionNode class
"""
import torch
import torch.nn as nn
import numpy as np
from .base import ActionNode

class IdxChangeNode(ActionNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


class ReshapeNode(ActionNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


class DimSwitchNode(ActionNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


class MapChangeNode(ActionNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


"""
ActionNode::IdxChangeNode class
"""

class ConcatNode(IdxChangeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)
        self.dim = gradfn._saved_dim
        # order is reversed as we use pop() when searching the backward
        self.prev_order2node = {}
        self.prev_order_count = 0
        self.in_channels = 0
        self.out_channels = 0
        self.pruned_count = 0
        self.order2idx = {}

    def add_prev(self, prev_node):
        if prev_node not in self.prev:
            self.prev.append(prev_node)
            self.prev_order2node[self.prev_order_count] = prev_node
            self.prev_order_count += 1
            self.in_channels += prev_node.out_channels
            self.out_channels = self.in_channels
            prev_node.add_next(self)

    def pass_idx(self, idx, group):
        for order, prev_node in self.prev_order2node.items():
            if not prev_node in group.prev_group.nodes:
                continue
            new_order = self.prev_order_count - order-1
            offset = 0
            for i in range(new_order):
                order_ori = self.prev_order_count - i - 1
                offset += self.prev_order2node[order_ori].out_channels
            self.order2idx[new_order] = idx + offset
            self.pruned_count +=1
        if self.pruned_count == self.prev_order_count:
            # concat the self.order2idx
            new_idx = torch.concat([self.order2idx[i] for i in range(self.prev_order_count)], dim=0)
            self.output_group.prune(new_idx)


class SplitNode(IdxChangeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)
        self.in_shape = gradfn._saved_self_sym_sizes
        self.in_channels = self.in_shape[1]
        self.out_channels = self.in_channels
        if gradfn._saved_dim < len(self.in_shape):
            self.dim = gradfn._saved_dim
        else:
            self.dim = gradfn._saved_dim - 18446744073709551616
        self.next_order2node = {}
        self.next_order_count = 0

    def add_next(self, next_node):
        if next_node not in self.next:
            self.next.append(next_node)
            self.next_order2node[self.next_order_count] = next_node
            self.next_order_count += 1

    def pass_idx(self, idx, group):
        length = self.out_channels // self.next_order_count
        idx_list = []
        for order in range(self.next_order_count):
            # get the idx, whose idx >= order * length and idx < (order + 1) * length:
            idx_list.append(idx[(idx >= order * length) & (idx < (order + 1) * length)] - order * length)

        for i, next_node in self.next_order2node.items():
            if isinstance(next_node, ActionNode):
                next_node.pass_idx(idx_list[i], None)
            else:
                next_node.prune(idx_list[i], self.dim)


class IndexSelectNode(IdxChangeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)
        self.in_shape = list(gradfn._saved_self_sym_sizes)
        for i, idx in enumerate(gradfn._saved_indices):
            if idx != None:
                self.idx = idx
                self.dim = i
        self.indices = (
                (slice(None),) * (self.dim)
                + (self.idx,)
                + (slice(None),) * (len(self.in_shape) - self.dim - 1)
        )
        self.out_shape = self.in_shape.copy()
        self.out_shape = self.out_shape[self.indices]
        self.in_channels = self.in_shape[1]
        self.out_channels = self.out_shape[1]


class SliceNode(IdxChangeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)
        info_list = []
        self.in_shape = list(gradfn._saved_self_sym_sizes)
        self.out_shape = self.in_shape.copy()
        self.dim = gradfn._saved_dim
        self.start = self._restore_idx(gradfn._saved_start)
        self.end = self._restore_idx(gradfn._saved_end)
        while gradfn.__class__.__name__ == "SliceBackward0":
            info_list.append(self._grad2info(grad))
            grad = grad.next_functions[0][0]
        for info in info_list:
            if info[3] != -1:
                self.dim = info[1]
                self.in_shape = info[0].copy()
                self.out_shape = info[0].copy()
                self.out_shape[self.dim] = info[3] - info[2]

    def _restore_idx(self, idx):
        return idx if idx <= 1000 else idx - 9223372036854775808

    def _grad2info(self, gradfn):
        in_shape = list(gradfn._saved_self_sym_sizes)
        dim = gradfn._saved_dim
        start = self._restore_idx(gradfn._saved_start)
        end = self._restore_idx(gradfn._saved_end)
        step = self._restore_idx(gradfn._saved_step)
        return [in_shape, dim, start, end, step]


"""
ActionNode::ReshapeNode class
"""

class FlattenNode(ReshapeNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        self.in_shape = list(gradfn._saved_self_sym_sizes)
        self.out_shape = [self.in_shape[0], np.prod(self.in_shape[1:])]
        self.in_channels = self.in_shape[1]
        self.out_channels = self.out_shape[1]

    def pass_idx(self, idx, group=None):
        fake_input = torch.zeros(self.in_shape)
        fake_input[:, idx, ::] = 1
        fake_output = fake_input.reshape(self.out_shape)
        new_idx = torch.where(fake_output == 1)[1]
        self.output_group.prune(new_idx)
        # self.input_group.prune(idx)


class UnsqueezeNode(ReshapeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


class SqueezeNode(ReshapeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


"""
ActionNode::DimSwitchNode class
"""


class TransposeNode(DimSwitchNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


class PermuteNode(DimSwitchNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)


class RearrangeNode(DimSwitchNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)



"""
ActionNode::MapChangeNode class
"""

class UpsampleNode(MapChangeNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)


class PoolNode(MapChangeNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)
        try:
            self.in_shape = list(gradfn._saved_self_sizes)
        except:
            self.in_shape = list(gradfn._saved_self.shape)
        self.in_channels = self.in_shape[1]
        self.out_channels = self.in_channels



class ExpandNode(MapChangeNode):
    def __init__(self, name: str, module, gradfn):
        super().__init__(name, module, gradfn)

class RepeatNode(MapChangeNode):
    def __init__(self, name: str, gradfn):
        super().__init__(name, None, gradfn)
        self.repeats = list(gradfn._saved_repeats)
        self.in_shape = list(gradfn._saved_self_sym_sizes)
        self.out_shape = self.in_shape.copy()
        try:
            self.dim = int(np.nonzero(np.asarray(self.repeats) - 1)[0])
        except:
            self.dim = 1
        self.out_shape[self.dim] = self.in_shape[self.dim] * self.repeats[self.dim]
        self.in_channels = self.in_shape[1]
        self.out_channels = self.out_shape[1]
