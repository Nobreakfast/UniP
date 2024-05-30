import torch
import torch.nn as nn
import abc
import logging

from unip.core.graph import name2grapher
from unip.core.group import name2grouper
from unip.core.algorithm import name2algorithm
from unip.mask.algorithm import name2pai
from unip.mask.unstructural import get_lw_sparsity, remove_mask
from unip.utils.data_type import DEVICE

logger = logging.getLogger("[Pruner:")


def name2pruner(name):
    if name == "OneShot":
        return OneShotPruner
    elif name == "PPaI":
        return PPaIPruner
    else:
        raise ValueError(
            f"Unsupported pruner name: {name}. \
            Please use 'OneShot' or 'PPaI'. \
            Or leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
        )


class BasePruner(abc.ABC):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
    ):
        self.model = model.to(DEVICE)
        self.example_input = example_input
        logger.info(f"{self.__class__.__name__}] Selected.")

    @abc.abstractmethod
    def save(self):
        pass


class StructuralPruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        grapher: str = "backward",
        grouper: str = "add",
        ignore_modules=None,
    ):
        super().__init__(model, example_input)
        self.ignore_modules = ignore_modules
        self.grapher = name2grapher(grapher)(model, example_input, ignore_modules)
        self.grouper = name2grouper(grouper)(model, example_input, self.grapher.graph)

    def prune(self):
        self._prune()

    @abc.abstractmethod
    def _prune(self):
        pass

    def plot(self, group=False, **kwargs):
        if group:
            self.grouper.plot(**kwargs)
        else:
            self.grapher.plot(**kwargs)

    def save(self, path: str = "./logs/", direct: bool = True):
        # TODO: save pruned model
        raise NotImplementedError


class OneShotPruner(StructuralPruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        grapher: str = "backward",
        grouper: str = "add",
        algorithm: str = "uniform",
        score: str = "l1",
        ratio=0.5,
        ignore_modules=None,
    ):
        super().__init__(model, example_input, grapher, grouper, ignore_modules)
        self.algorithm = name2algorithm(algorithm)(self.grouper.group, ratio, score)

    def _prune(self):
        self.algorithm.prune()


class PPaIPruner(StructuralPruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        pai: str = "synflow",
        ratio: float = 0.5,
        algorithm: str = "lw",
        score: str = "l1",
        ignore_modules=None,
    ):
        super().__init__(model, example_input, ignore_modules)
        self.ratio = ratio
        self.score = score
        lw_ratio = self.get_lw_ratio(pai)
        self.algorithm = name2algorithm(algorithm)(
            self.grouper.group, lw_ratio, score=score
        )

    def get_lw_ratio(self, pai):
        name2pai(pai)(self.model, self.example_input, self.ratio, DEVICE)
        remove_mask(self.model)
        self.model.zero_grad()
        return get_lw_sparsity(self.model)

    def _prune(self):
        self.algorithm.prune()


class MaskPruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
    ):
        super().__init__(model, example_input)

    def save(self):
        # TODO: save pruned model
        raise NotImplementedError


class MaskStructuralPruner(MaskPruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
    ):
        super().__init__(model, example_input)


class MaskUnstructuralPruner(MaskPruner):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
    ):
        super().__init__(model, example_input)
