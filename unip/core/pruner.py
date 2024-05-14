from unip.core.graph import BackwardGrapher, BaseGrapher

import torch
import torch.nn as nn
import abc


class BasePruner(abc.ABC):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        algorithm: str = "Uniform",
    ):
        self.model = model
        self.example_input = example_input
        self.algorithm = algorithm
        grapher = BackwardGrapher(model, example_input)
        self.graph = grapher.graph
