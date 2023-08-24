import os

import torch
import numpy as np

from unip.utils.deploy import *


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        elif isinstance(item, torch.Tensor):
            flattened.append(item.detach().numpy())
        else:
            flattened.append(item)
    return flattened


def compare_two_results(result1, result2):
    result1 = flatten_list(result1)
    result2 = flatten_list(result2)
    for r1, r2 in zip(result1, result2):
        assert np.allclose(r1, r2, atol=1e-5)
    return True


class example_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.conv2_1 = torch.nn.Conv2d(4, 5, 3)
        self.conv2_2 = torch.nn.Conv2d(4, 5, 3)
        self.conv3 = torch.nn.Conv2d(5, 6, 3)
        self.conv4 = torch.nn.Conv2d(6, 7, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x21 = self.conv2_1(x1)
        x22 = self.conv2_2(x1)
        x = self.conv3(x21)
        x = self.conv4(x)
        return [x1, [x21, x22], x]


def test_output2name():
    model = example_model()
    example_input = torch.randn(1, 3, 224, 224)
    output = model(example_input)
    output_names = output2name(output)
    assert output_names == ["output_0", "output_1_0", "output_1_1", "output_2"]


def test_torch2onnx_verify_onnx():
    model = example_model()
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    torch2onnx(model, example_input, "tests/.example_model.onnx")
    onnx_model = verify_onnx("tests/.example_model.onnx")


def test_inference_onnx():
    model = example_model()
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    output_torch = model(example_input)
    torch2onnx(model, example_input, "tests/.example_model.onnx")
    output_onnx = inference_onnx(
        "tests/.example_model.onnx", example_input.detach().numpy()
    )
    assert compare_two_results(output_torch, output_onnx)


if __name__ == "__main__":
    test_output2name()
    test_torch2onnx_verify_onnx()
