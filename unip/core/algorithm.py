import torch
import torch.nn as nn

import abc

from .group import *
from .node import *
from .score import *


class BaseAlgo(abc.ABC):
    def __init__(self, groups, key2node):
        self.groups = groups
        self.key2node = key2node
        self.non_pruneable_group = self.groups.copy()
        self.prunable_groups = []
        for g in self.groups:
            if not g.is_prunable:
                continue
            self.prunable_groups.append(g)
            self.non_pruneable_group.remove(g)

    def prune(self):
        for n in self.key2node.values():
            n.prune()
        return True

    @abc.abstractmethod
    def run(self, ratio):
        pass


class RatioAlgo(BaseAlgo):
    def __init__(self, groups, key2node, score_fn):
        super().__init__(groups, key2node)
        self.score_fn = score_fn

    def get_prune_idx(self, group, ratio):
        length = group.length
        length_reduce = group.length_reduce
        round_to = group.round_to * length_reduce
        split = group.next.split
        if length == []:
            return []
        num_toprune = int(length * ratio / round_to) * round_to // split
        if num_toprune == length:
            num_toprune -= round_to
        num_toprune = num_toprune // length_reduce

        score = self.score_fn(group)
        # score.shape = [length] => [length // split // length_reduce, -1]
        score = score.reshape(length // split // length_reduce, -1)
        score = score.sum(dim=1)
        # find the topk index
        _, tmp_prune_idx = torch.topk(score, num_toprune)

        tmp_prune_idx = torch.concat(
            [
                tmp_prune_idx + i * length // split // length_reduce
                for i in range(split * length_reduce)
            ]
        )
        prune_idx = tmp_prune_idx.tolist()
        if prune_idx != []:
            prune_idx.sort()
        return prune_idx

    def _get_prune_idx(self, group, ratio):
        prune_idx = group.has_prune_idx()
        if prune_idx != None:
            return prune_idx
        return self.get_prune_idx(group, ratio)

    def run(self, ratio):
        self.group2ratio = self.get_group2ratio(ratio)
        for g in self.prunable_groups:
            prune_idx = self._get_prune_idx(g, self.group2ratio[g])
            g.add_prune_idx(prune_idx)
        search_list = self.non_pruneable_group.copy()
        for g in self.non_pruneable_group:
            g = search_list.pop(0)
            prune_idx = self._get_prune_idx(g, self.group2ratio[g])
            g.add_prune_idx(prune_idx)
            if not g.pruned:
                search_list.append(g)

    @abc.abstractmethod
    def get_group2ratio(self, ratio):
        pass


class UniformAlgo(RatioAlgo):
    def __init__(self, groups, key2node, score_fn=weight_sum_l1_out):
        # def __init__(self, groups, key2node, score_fn=rand):
        super().__init__(groups, key2node, score_fn)

    def get_group2ratio(self, ratio):
        group2ratio = {}
        for g in self.groups:
            group2ratio[g] = ratio
        return group2ratio


class RandomAlgo(RatioAlgo):
    """Random Algorithm: this is only used for testing the pruning process."""

    def __init__(self, groups, key2node, score_fn=rand):
        super().__init__(groups, key2node, score_fn)

    def get_group2ratio(self, ratio):
        group2ratio = {}
        for g in self.groups:
            group2ratio[g] = torch.rand(1).item()
        return group2ratio


class GlobalAlgo(BaseAlgo):
    def __init__(self, groups, key2node):
        super().__init__(groups, key2node)

    def run(self, ratio):
        return []
