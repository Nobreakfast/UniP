import torch
import torch.nn as nn

import abc

from ..utils.data_type import DIM_IN, DIM_OUT, IDX_IN, IDX_OUT
from .node import *


class BaseGroup(abc.ABC):
    def __init__(self, nodes):
        self.nodes = nodes
        self.is_prunable, self.is_only_nonprune = self._is_prunable()
        self.pruned = False

    def update_nodes(self):
        for node in self.nodes:
            node.add_group(self)

    def add_prune_idx(self, prune_idx, dim):
        if self.pruned:
            return
        for node in self.nodes:
            ret = node.add_prune_idx(prune_idx, dim)
            if not ret:  # False means this node is not pruned
                break
        self.pruned = ret

    def get_index_length(self):
        length = []
        for n in self.nodes:
            if isinstance(n, BundleParamNode):
                continue
            if isinstance(n, (InOutNode, RemapNode)):
                length.append(n.out_channels)

        length = list(set(length))
        assert len(length) <= 1, f"expected len(length) of 1 or 0, got {length}"
        return length if len(length) == 0 else length[0]

    def get_round_to(self):
        round_to = [1]
        for n in self.nodes:
            round_to.append(n.round_to)
        return max(round_to)

    def get_split(self):
        split = [1]
        for n in self.nodes:
            split.append(n.split)
        return max(split)

    def _is_prunable(self):
        self.is_prunable = True
        self.is_only_nonprune = True
        for n in self.nodes:
            if not n.is_prunable:
                self.is_prunable = False
            else:
                self.is_only_nonprune = False
        return self.is_prunable, self.is_only_nonprune

    def has_prune_idx(self):
        for n in self.nodes:
            prune_idx = n.prune_idx[IDX_OUT]
            if prune_idx != None:
                return prune_idx
        return None

    def get_length_reduce(self):
        for n in self.nodes:
            if isinstance(n, LastLinearNode):
                return 4
        return 1


class CurrentGroup(BaseGroup):
    def __init__(self, nodes):
        super(CurrentGroup, self).__init__(nodes)
        self.next = self.get_next_group()
        self.length = self.get_index_length()
        self.length_reduce = self.get_length_reduce()
        self.round_to = self.next.round_to
        self.split = self.next.split
        self.update_nodes()
        # remove hasdummy
        # if self.next.hasdummy:
        #     self.pruned = True
        #     for n in self.nodes:
        #         n.add_prune_idx([], IDX_OUT)

    def get_next_group(self):
        nodes = []
        for n in self.nodes:
            nodes.extend(n.next)
            nodes.extend(n.next_key)
        nodes = list(set(nodes))
        return NextGroup(nodes)

    def add_prune_idx(self, prune_idx):
        super().add_prune_idx(prune_idx, IDX_OUT)
        # self.next.add_prune_idx(prune_idx)


# DONE: check this is useful, because we do not use this as main group anymore
#       Now, this will only be used to capture the information of next group
class NextGroup(BaseGroup):
    def __init__(self, nodes):
        super(NextGroup, self).__init__(nodes)
        self.round_to = self.get_round_to()
        self.split = self.get_split()

    # remove hasdummy
    # @property
    # def hasdummy(self):
    #     for node in self.nodes:
    #         if isinstance(node, DummyNode):
    #             return True
    #     return False

    # def add_prune_idx(self, prune_idx):
    #     super().add_prune_idx(prune_idx, IDX_IN)
