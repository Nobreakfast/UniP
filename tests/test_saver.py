import torch
import torch.nn as nn
import torchvision.models as models
import unip
from unip.core.pruner import BasePruner
from unip.core.algorithm import UniformAlgo
from unip.core.node import *
from unip.utils.evaluation import *
from unip.utils.saver import *

from model.example import ExampleModel

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_with_example():
    print("=" * 20, "test_model_with_example", "=" * 20)
    model = ExampleModel()
    example_input = torch.randn(1, 3, 4, 4, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(model, example_input, algorithm=UniformAlgo)
    BP.algorithm.run(0.3)
    BP.prune()
    cal_flops(model, example_input, device)
    assert model(example_input).shape == out1.shape
    save_model(model, ".example_model.pt")
    model = load_model(".example_model.pt")
    assert model(example_input).shape == out1.shape
    os.system("rm .example_model.pt")


def test_model_dict_with_example():
    print("=" * 20, "test_model_dict_with_example", "=" * 20)
    model = ExampleModel()
    example_input = torch.randn(1, 3, 4, 4, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(model, example_input, algorithm=UniformAlgo)
    BP.algorithm.run(0.3)
    BP.prune()
    cal_flops(model, example_input, device)
    assert model(example_input).shape == out1.shape
    save_model_dict(model, BP.key2node, ".example_model.pth")
    model = ExampleModel()
    model = load_model_dict(model, ".example_model.pth")
    assert model(example_input).shape == out1.shape
    os.system("rm .example_model.pth")


if __name__ == "__main__":
    test_model_with_example()
    test_model_dict_with_example()
