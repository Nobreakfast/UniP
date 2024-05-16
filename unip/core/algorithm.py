import abc
import math

import torch
import torch.nn as nn
from einops import rearrange
import logging

from unip.core.graph import BackwardGrapher
from unip.core.group import AddGrouper
from unip.core.node import *
from unip.core.score import name2scorefn
from unip.utils.data_type import *

logger = logging.getLogger("[Algorithm:")


def name2algorithm(name):
    if name == "uniform":
        return UniformAlgorithm
    elif name == "lw":
        return LayerWiseAlgorithm
    else:
        raise ValueError(
            f"Unsupported algorithm name: {name}. \
            Please use 'uniform'. \
            Or leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
        )


class BaseAlgorithm(abc.ABC):
    def __init__(self, groups: list):
        self.groups = groups
        logger.info(f"{self.__class__.__name__}] Selected.")

    def prune(self):
        self._prune()

    @abc.abstractmethod
    def _prune(self):
        pass


class GlobalScoreAlgorithm(BaseAlgorithm):
    def __init__(self, groups: list):
        super().__init__(groups)

    def _prune(self):
        pass


class LayerWiseAlgorithm(BaseAlgorithm):
    def __init__(self, groups: list, lw_ratio: dict, score: str = "l1"):
        super().__init__(groups)
        self.groups = groups
        self.lw_ratio = lw_ratio
        self.score_fn = name2scorefn(score)

    def _prune(self):
        for group in self.groups:
            idx = self._get_prune_idx(group)
            if idx is not None:
                logger.info(
                    f"{self.__class__.__name__}] Pruning {idx.numel()} nodes in group: {[n.name for n in group.nodes]}"
                )
            else:
                logger.info(
                    f"{self.__class__.__name__}] Skip group: {[n.name for n in group.nodes]}"
                )
            group.prune(idx)

    def _get_prune_idx(self, group):
        ratio = []
        for node in group.nodes:
            if node.name not in self.lw_ratio.keys():
                continue
            ratio.append(self.lw_ratio[node.name])
        ratio = torch.tensor(ratio).mean()
        # TODO: 1. calculate the score of prunable index in same group
        if not group.prunable or group.length == 1:
            return None
        score = self.score_fn(group.prunable_param.values(), group.length)
        # TODO: 2. find the threshold of the score
        th = torch.quantile(score, ratio)
        # TODO: 3. prune the lowest score index base on the ratio
        saved_idx = torch.where(th < score)[0].int()
        return saved_idx


class UniformAlgorithm(LayerWiseAlgorithm):
    def __init__(self, groups: list, ratio: float = 0.5, score: str = "l1"):
        self.ratio = ratio
        lw_ratio = {}
        for group in groups:
            for node in group.nodes:
                lw_ratio[node.name] = ratio
        super().__init__(groups, lw_ratio, score)


if __name__ == "__main__":

    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(8)
            self.p1 = nn.Parameter(torch.randn(1, 1, 1, 4))
            self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
            self.conv3 = nn.Conv2d(4, 8, 3, 1, 1)
            self.bn23 = nn.BatchNorm2d(16)
            self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
            self.bn4 = nn.BatchNorm2d(16)
            self.conv5_1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.conv5_2 = nn.Conv2d(16, 16, 3, 1, 1)
            self.identity = nn.Identity()
            self.bn5i = nn.BatchNorm2d(16)
            self.bn45 = nn.BatchNorm2d(32)
            self.fcp = nn.Linear(32, 32)
            self.conv6 = nn.Conv2d(32, 1, 3, 1, 1)
            self.bn6 = nn.BatchNorm2d(1)
            self.p6 = nn.Parameter(torch.randn(1, 1, 1, 1))
            # self.pool = nn.MaxPool2d(2, 2)
            self.pool = nn.AvgPool2d(2, 2)
            self.flat = nn.Flatten()
            self.fc = nn.Linear(4, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = x + self.p1
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.conv2(x1)
            x2 = self.conv3(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn23(x)
            identity = self.identity(x)
            x1 = self.conv4(x)
            x1 = self.bn4(x1)
            x2 = self.conv5_1(x)
            x2 = self.conv5_2(x2 + identity)
            x2 = self.bn5i(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn45(x)  # (1, 32, 4, 4)
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.fcp(x)  # (1, 4, 4, 32) -> (1, 4, 4, 32)
            x = rearrange(x, "b (h w) c -> b c h w", h=4)
            x = self.conv6(x)  # (1, 32, 4, 4)
            x = self.bn6(x)
            x = torch.nn.functional.relu(x)
            x = x * self.p6
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x)
            return x

    class TestModel2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(8)
            self.conv3 = nn.Conv2d(8, 8, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(8)
            self.conv2_1 = nn.Conv2d(8, 8, 3, 1, 1)
            self.bn2_1 = nn.BatchNorm2d(8)
            self.conv3_1 = nn.Conv2d(8, 8, 3, 1, 1)
            self.bn3_1 = nn.BatchNorm2d(8)
            self.flat = nn.Flatten()
            self.flat2 = nn.Flatten()
            self.fc = nn.Linear(128, 10)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x1 = self.conv2(x)
            x1 = self.bn2(x1)
            x2 = self.conv3(x)
            x2 = self.bn3(x2)

            # x = torch.cat([x1, x2], dim=1)
            # x = self.flat(x)
            # x1 = self.fc(x)

            x = torch.cat([x1, x2], dim=1)
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.conv2_1(x1)
            x1 = self.bn2_1(x1)
            x2 = self.conv3_1(x2)
            x2 = self.bn3_1(x2)
            x1 = self.flat(x1)
            x2 = self.flat2(x2)
            x1 = self.fc(x1)
            x2 = self.fc2(x2)
            return x1, x2

    model = TestModel2()
    example_input = torch.randn(1, 3, 4, 4)
    graph = BackwardGrapher(model, example_input).graph
    groups = AddGrouper(model, example_input, graph).group
    for i, group in enumerate(groups):
        print(f"Group [{i}]: {[n.name for n in group.nodes]}")
        print(
            f"Length: {group.length}; Prunable: {group.prunable}; Next Group [{i}]: {[n.name for n in group.group_next.nodes] if group.group_next else None}"
        )
    alogorithm = UniformAlgorithm(groups, 0.7)
    print(model)
    print(model(example_input))
