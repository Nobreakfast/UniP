import torch
import torch.nn as nn

import abc

from .group import *
from .node import *


class BaseAlgo(abc.ABC):
    def __init__(self, groups, key2node):
        self.groups = groups
        self.key2node = key2node
        # self.run(0.3)

    def prune(self):
        for n in self.key2node.values():
            n.prune()
        return True

    @abc.abstractmethod
    def get_prune_idx(self, group, ratio):
        pass

    def _get_prune_idx(self, group, ratio):
        prune_idx = group.has_prune_idx()
        if prune_idx != None:
            return prune_idx
        return self.get_prune_idx(group, ratio)

    def run(self, ratio):
        search_list = self.groups.copy()
        for g in self.groups:
            if not g.is_prunable:
                continue
            prune_idx = self._get_prune_idx(g, ratio)
            g.add_prune_idx(prune_idx)
            search_list.remove(g)
        while search_list != []:
            g = search_list.pop(0)
            prune_idx = self._get_prune_idx(g, ratio)
            g.add_prune_idx(prune_idx)
            if not g.pruned:
                search_list.append(g)


class RandomAlgo(BaseAlgo):
    def __init__(self, groups, key2node):
        super().__init__(groups, key2node)

    def get_prune_idx(self, group, ratio):
        return []


class UniformAlgo(BaseAlgo):
    def __init__(self, groups, key2node):
        super().__init__(groups, key2node)

    def get_prune_idx(self, group, ratio):
        length = group.length
        round_to = group.next.round_to
        split = group.next.split
        if length == []:
            return []
        num_toprune = int(length * ratio / round_to) * round_to // split
        prune_idx = torch.randperm(length // split)[:num_toprune]
        prune_idx = torch.concat(
            [prune_idx + i * length // split for i in range(split)]
        ).tolist()
        return prune_idx
