import torch
import torch.nn as nn
import abc
import time
from einops import rearrange
import logging

from unip.core.graph import BackwardGrapher
from unip.core.node import *
from unip.utils.data_type import *
from unip.utils.plot import plot_group

logger = logging.getLogger("[Group:")


def name2grouper(name):
    if name == "add":
        return AddGrouper
    else:
        raise ValueError(
            f"Unsupported grouper name: {name}. \
            Please use 'add'. \
            Or leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
        )


class BasePruneGroup(abc.ABC):
    def __init__(self):
        self.nodes = set()
        self.prunable_param = {}  # node to param
        self.prune_idx = []
        self.length = set()
        self.nodes_type = set()
        self.prunable = True
        self.outputnode = False

    @abc.abstractmethod
    def prune(self, idx):
        pass

    @abc.abstractmethod
    def update_info(self):
        pass

    @abc.abstractmethod
    def add_node(self, node):
        pass


class OutputPruneGroup(BasePruneGroup):
    def __init__(self):
        super().__init__()
        self.group_next = None

    def add_node(self, node):
        # 1. add node to the group and add the group to the node
        self.nodes.add(node)
        node.add_group(self, "output")
        # 2. check the prunable param
        if isinstance(node, (InOutNode, NormNode, BundleNode)):
            # TODO: if the pruning dim is not 0 and 1
            self.prunable_param[node] = node.prunable_param
            # 3. add the length of the node to the group
            self.length.add(node.out_channels)
        elif isinstance(node, (ActionNode, IgnoreNode)):
            self.length.add(node.out_channels)
            self.prunable = False

    def prune(self, prune_idx):
        if prune_idx is None:
            return
        for node in self.prunable_param.keys():
            node.prune(prune_idx)
        self.group_next.prune(prune_idx)

    def update_info(self):
        self._update_info()
        self._get_group_next()
        self.group_next.update_info()
        if self.group_next.outputnode:
            self.prunable = False

    def _update_info(self):
        # TODO: calculate the round
        length = list(self.length)
        if len(length) == 1:
            self.length = length[0]
        elif length == []:
            self.length = 1
        else:
            # FIXME: not correct
            try:
                self.length = max(length)
            except:
                self.length = 1
        if self.length == 1:
            self.prunable = False

    def _get_group_next(self):
        self.group_next = InputPruneGroup()
        self.group_next.prev_group = self
        tmp_checkout_list = [n for n in self.nodes]
        while tmp_checkout_list:
            node = tmp_checkout_list.pop(0)
            for next_node in node.next:
                if not isinstance(
                    next_node, (DummyNode, InOutNode, ActionNode, ActivationNode)
                ):
                    continue
                self.group_next.add_node(next_node)


class InputPruneGroup(BasePruneGroup):
    def __init__(self):
        super().__init__()
        self.prev_group = None

    def add_node(self, node):
        # 1. add node to the group and add the group to the node
        self.nodes.add(node)
        node.add_group(self, "input")
        # 2. check the prunable param
        if isinstance(node, (InOutNode)):
            self.prunable_param[node] = node.prunable_param
            # 3. add the length of the node to the group
            self.length.add(node.in_channels)
        elif isinstance(node, ActionNode):
            self.length.add(node.in_channels)
        elif isinstance(node, OutputNode):
            self.length.add(node.in_channels)
            self.prunable = False
            self.outputnode = True
        elif isinstance(node, ActivationNode):
            self.length.add(node.in_channels)

    def prune(self, prune_idx):
        checkin_list = [n for n in self.nodes]
        for node in self.prunable_param.keys():
            node.prune(prune_idx, dim=1)
            checkin_list.remove(node)
        for node in checkin_list:
            node.pass_idx(prune_idx, self)

    def update_info(self):
        self._update_info()

    def _update_info(self):
        length = list(self.length)
        if len(length) == 1:
            self.length = length[0]
        elif length == []:
            self.length = 1
        else:
            # FIXME: not correct
            # print([n.name for n in self.nodes])

            self.length = max(length)

        if self.length == 1:
            self.prunable = False


class BaseGrouper(abc.ABC):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        graph: dict,
    ):
        self.model = model
        self.example_input = example_input
        self.graph = graph
        self.groups = None

    @property
    def group(self):
        if self.groups is None:
            logger.info(f"{self.__class__.__name__}] Finding groups...")
            self.groups = self._find_groups()
            self._update_info()
            logger.info(f"{self.__class__.__name__}] Found groups.")
        return self.groups

    @abc.abstractmethod
    def _find_groups(self):
        pass

    def _update_info(self):
        for group in self.groups:
            group.update_info()

    def plot(self, display=True, save_path=None):
        if save_path is None:
            save_path = f"logs/plot/fig_{time.time()}"
            logger.info(f"{self.__class__.__name__}] Save plot to {save_path}")
        plot_group(self.groups, display=display, save_path=save_path)


class AddGrouper(BaseGrouper):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        graph: dict,
    ):
        super().__init__(model, example_input, graph)

    def _search_ininnode(self, node, checked_list):
        all_inin_nodes = [node]
        # 1. search the next nodes' nodes, if exist a InInNode
        checkout_list = [n for n in node.next]
        while checkout_list:
            next_node = checkout_list.pop(0)
            if isinstance(next_node, InInNode):
                if next_node in checked_list:
                    continue
                all_inin_nodes.append(next_node)
                checked_list.append(next_node)
                tmp_all_inin_nodes, checked_list = self._search_ininnode(
                    next_node, checked_list
                )
                all_inin_nodes += tmp_all_inin_nodes
            elif not isinstance(next_node, InOutNode):
                checkout_list += next_node.next

        # 2. search the prev nodes' next nodes, if exist a InInNode
        checkout_list = [n for n in node.prev]
        while checkout_list:
            prev_node = checkout_list.pop(0)
            for prev_next_node in prev_node.next:
                if isinstance(prev_next_node, InInNode):
                    if prev_next_node in checked_list:
                        continue
                    all_inin_nodes.append(prev_next_node)
                    checked_list.append(prev_next_node)
                    tmp_all_inout_nodes, checked_list = self._search_ininnode(
                        prev_next_node, checked_list
                    )
                    all_inin_nodes += tmp_all_inout_nodes
            if isinstance(prev_node, InInNode):
                if prev_node in checked_list:
                    continue
                all_inin_nodes.append(prev_node)
                checked_list.append(prev_node)
                tmp_all_inout_nodes, checked_list = self._search_ininnode(
                    prev_node, checked_list
                )
                all_inin_nodes += tmp_all_inout_nodes

            elif not isinstance(prev_node, InOutNode):
                checkout_list += prev_node.prev
        return all_inin_nodes, checked_list

    def _find_groups(self):
        groups = []
        checkin_list = []
        checkout_list = []
        # add checkin list
        for name, node in self.graph.items():
            if isinstance(node, InInNode):
                checkin_list.append(node)
        # search checkin_list
        while checkin_list:
            node = checkin_list.pop(0)
            checkout_list.append(node)
            # search the InOutNode arround the node
            all_inin_nodes, _ = self._search_ininnode(node, [node])
            all_inin_nodes = list(set(all_inin_nodes))
            # search the all_inin_node, group the nodes
            group = OutputPruneGroup()
            for inin_node in all_inin_nodes:
                group.add_node(inin_node)
                checkout_list.append(inin_node)
                if inin_node in checkin_list:
                    checkin_list.remove(inin_node)
                tmp_checkout_list = [n for n in inin_node.prev]
                while tmp_checkout_list:
                    inin_prev_node = tmp_checkout_list.pop(0)
                    group.add_node(inin_prev_node)
                    checkout_list.append(inin_prev_node)
                    if not isinstance(inin_prev_node, (InOutNode, ActionNode)):
                        tmp_checkout_list += inin_prev_node.prev
            groups.append(group)
            logger.info(
                f"{self.__class__.__name__}] Add Group having InInNode: {[n.name for n in group.nodes]}"
            )
        for node in self.graph.values():
            if node in checkout_list:
                continue
            if isinstance(node, InOutNode):
                group = OutputPruneGroup()
                group.add_node(node)
                checkout_list.append(node)
                # search next, if it is a NormNode, add it to the group
                tmp_checkout_list = [n for n in node.next]
                while tmp_checkout_list:
                    next_node = tmp_checkout_list.pop(0)
                    if next_node in checkout_list:
                        continue
                    if isinstance(next_node, NormNode):
                        group.add_node(next_node)
                        checkout_list.append(next_node)
                        tmp_checkout_list += next_node.next
                    # elif isinstance(next_node, DimChangeNode):
                    #     group.add_node(next_node)
                    #     checkout_list.append(next_node)
                    #     tmp_checkout_list += next_node.prev
                groups.append(group)
                logger.info(
                    f"{self.__class__.__name__}] Add Group Having Modules: {[n.name for n in group.nodes]}"
                )
            elif isinstance(node, NormNode):
                group = OutputPruneGroup()
                group.add_node(node)
                checkout_list.append(node)
                # search prev, if it is a InOutNode, add it to the group
                tmp_checkout_list = [n for n in node.prev]
                while tmp_checkout_list:
                    inin_prev_node = tmp_checkout_list.pop(0)
                    group.add_node(inin_prev_node)
                    checkout_list.append(inin_prev_node)
                    if isinstance(inin_prev_node, NormNode):
                        tmp_checkout_list += inin_prev_node.prev
                groups.append(group)
                logger.info(
                    f"{self.__class__.__name__}] Add Group Having Modules: {[n.name for n in group.nodes]}"
                )
            elif isinstance(node, (ActionNode, ActivationNode)):
                group = OutputPruneGroup()
                group.add_node(node)
                checkout_list.append(node)
                groups.append(group)
                logger.info(
                    f"{self.__class__.__name__}] Add Group Having Others: {[n.name for n in group.nodes]}"
                )
        return groups


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

    model = TestModel()
    example_input = torch.randn(1, 3, 4, 4)
    graph = BackwardGrapher(model, example_input).graph

    grouper = AddGrouper(model, example_input, graph)
    groups = grouper.group
    for i, group in enumerate(groups):
        print(f"Group [{i}]: {[n.name for n in group.nodes]}")
        print(
            f"Length: {group.length}; Prunable: {group.prunable}; Next Group [{i}]: {[n.name for n in group.group_next.nodes] if group.group_next else None}"
        )
    print("Done!")
