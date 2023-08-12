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
        length_reduce = group.length_reduce
        round_to = group.round_to * length_reduce
        split = group.next.split
        if length == []:
            return []
        num_toprune = int(length * ratio / round_to) * round_to // split
        if num_toprune == length:
            return []
        # if length_reduce == 4:
        #     for n in group.nodes:
        #         if "to_qkv" in n.name:
        #             print("length_reduce == 8")
        #         if "to_out" in n.name:
        #             print("length_reduce == 8")
        tmp_prune_idx = torch.randperm(length // split // length_reduce)[
            : num_toprune // length_reduce
        ]
        # tmp_prune_idx = tmp_prune_idx * length_reduce
        tmp_prune_idx = torch.concat(
            [
                tmp_prune_idx + i * length // split // length_reduce
                for i in range(split * length_reduce)
            ]
        )
        prune_idx = tmp_prune_idx.tolist()
        # prune_idx = []
        # for i in range(length_reduce):
        #     prune_idx.append(tmp_prune_idx + i)
        # prune_idx = torch.concat(prune_idx).tolist()
        if prune_idx != []:
            prune_idx.sort()
        return prune_idx
        # return tmp_prune_idx.tolist()
