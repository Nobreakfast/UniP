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
        # update the split to prunable_groups
        # FIXME: may have bugs
        search_list = self.non_pruneable_group.copy()
        while search_list != []:
            g = search_list.pop(0)
            if g.split == 1:
                continue
            for n in g.nodes:
                if not isinstance(n, (RemapNode, ReshapeNode)):
                    continue
                if n.prev_key[0].group.is_prunable:
                    n.prev_key[0].group.length_reduce = g.split
                else:
                    n.prev_key[0].group.split = g.split
                    search_list.append(n.prev_key[0].group)

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
        self.score_fn = name2score(score_fn)

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
        if num_toprune == 0:
            return []
        score = self.score_fn(group)
        # score.shape = [length] => [length // split // length_reduce, -1]
        score = score.reshape(length // split // length_reduce, -1)
        score = score.sum(dim=1)
        # find the topk index
        _, tmp_prune_idx = torch.topk(score, num_toprune)

        if split == 4:
            print("debug")
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
        # for g in self.non_pruneable_group:
        while search_list != []:
            g = search_list.pop(0)
            prune_idx = self._get_prune_idx(g, self.group2ratio[g])
            g.add_prune_idx(prune_idx)
            if not g.pruned:
                search_list.append(g)

    @abc.abstractmethod
    def get_group2ratio(self, ratio):
        pass


class UniformRatio(RatioAlgo):
    def __init__(self, groups, key2node, score_fn="weight_sum_l1_out"):
        super().__init__(groups, key2node, score_fn)

    def get_group2ratio(self, ratio):
        group2ratio = {}
        for g in self.groups:
            group2ratio[g] = ratio
        return group2ratio


class MTURatio(RatioAlgo):
    def __init__(
        self, groups, key2node, score_fn="weight_sum_l1_out", MTU: dict = None
    ):
        super().__init__(groups, key2node, score_fn)
        self.MTU = MTU

    def get_group2ratio(self, ratio):
        group2ratio = {}
        for g in self.groups:
            group2ratio[g] = ratio * self.get_group_tags_ratio(g)
        return group2ratio

    def get_group_tags_ratio(self, group):
        tags = []
        ratio = []
        for n in group.nodes:
            tags.extend(n.tags)
        tags = list(set(tags))
        for tag in tags:
            if tag not in self.MTU.keys():
                continue
            ratio.append(float(self.MTU[tag]))
        if len(ratio) == 0:
            return 1.0
        return torch.mean(torch.tensor(ratio))


class RandomRatio(RatioAlgo):
    """Random Algorithm: this is only used for testing the pruning process."""

    def __init__(self, groups, key2node, score_fn="rand"):
        super().__init__(groups, key2node, score_fn)

    def get_group2ratio(self, ratio=0.99):
        group2ratio = {}
        for g in self.groups:
            group2ratio[g] = torch.rand(1).item() * ratio
        return group2ratio


class GlobalAlgo(BaseAlgo):
    def __init__(self, groups, key2node):
        super().__init__(groups, key2node)

    def run(self, ratio):
        return []


def name2algo(name):
    return globals()[name]
