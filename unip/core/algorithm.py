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


class MMMTURatio(RatioAlgo):
    def __init__(
        self, groups, key2node, score_fn="weight_sum_l1_out", MMMTU: dict = None
    ):
        super().__init__(groups, key2node, score_fn)
        self.MMMTU = MMMTU

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
            if tag not in self.MMMTU.keys():
                continue
            ratio.append(float(self.MMMTU[tag]))
        if len(ratio) == 0:
            return 1.0
        return torch.mean(torch.tensor(ratio))


class AdaptiveMMU(MMMTURatio):
    def __init__(
        self,
        groups,
        key2node,
        score_fn="weight_sum_l1_out",
        model=None,
        example_input=None,
    ):
        super().__init__(groups, key2node, score_fn, {})
        self.MMMTU = self.calculate_MMMTU(model, example_input)

    def calculate_MMMTU(self, model, example_input) -> dict:
        tag2nodes = self.get_different_input_nodes()
        synflow_input = _generate_synflow_input(example_input)
        from unip.utils.evaluation import get_data
        from unip.core.pruner import sum_output

        fn, model, synflow_input = get_data(model, synflow_input, "cpu")
        model.train()
        out, _ = sum_output(fn(model, synflow_input))
        out.backward()

        tag2score = {}
        for k, v in tag2nodes.items():
            tag2score[k] = self.synflow_score(v)

        min_value = min(tag2score.values())
        for k, v in tag2score.items():
            # we don't want to change the ratio too much
            tag2score[k] = max(1 / (torch.log(v / min_value) + 1), 0.8)
        return tag2score

    def synflow_score(self, list):
        score = 0.0
        count = 0
        for n in list:
            count += n.module.weight.numel()
            score += torch.sum(
                n.module.weight.data.abs() * n.module.weight.grad.data.abs()
            )
        score /= count
        return score

    def get_fusion_nodes(self):
        tags_in = set()
        fusion_nodes = []
        for g in self.groups:
            for n in g.nodes:
                if not isinstance(n, (RemapNode, InInNode)):
                    continue
                input_tags, _ = n.get_tags_info()
                if len(input_tags) == 1:
                    continue
                tags_in.update(input_tags)
                if n.prev_has_one_tag():
                    fusion_nodes.append(n)
        return fusion_nodes, list(tags_in)

    def get_different_input_nodes(self) -> dict:
        fusion_nodes, tags_in = self.get_fusion_nodes()
        tag2nodes = {}
        for tag in tags_in:
            tag2nodes[tag] = []
        for n in fusion_nodes:
            for prev in n.prev_key:
                for prev_g_node in prev.group.nodes:
                    if not isinstance(prev_g_node, InOutNode):
                        continue
                    for tag in prev_g_node.get_tags_info()[0]:
                        tag2nodes[tag].append(prev_g_node)
        return tag2nodes


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


def _generate_synflow_input(data):
    if isinstance(data, torch.Tensor):
        return torch.ones_like(data)
    elif isinstance(data, dict):
        return {k: _generate_synflow_input(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_generate_synflow_input(v) for v in data]
    else:
        raise Exception("unsupport type: {}".format(type(data)))
