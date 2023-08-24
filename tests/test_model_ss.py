import torch
import torch.nn as nn
import torchvision.models as models

import unip
from unip.core.pruner import BasePruner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prune_ratio = torch.rand(1)


def prune(model):
    example_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    BP = BasePruner(model, example_input, "UniformRatio")
    BP.algorithm.run(prune_ratio)
    BP.prune()
    model(example_input)


if __name__ == "__main__":
    pass
