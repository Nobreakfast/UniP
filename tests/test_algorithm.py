import torch
import torch.nn as nn
import torchvision.models as models
import unip
from unip.core.pruner import BasePruner
from unip.core.node import *
from unip.utils.evaluation import *

from model.example import ExampleModel
from model.example import ShuffleAttention
from model.radarnet import RCNet
from model.backbone.vision.mobilevit_modules.mobilevit import (
    mobilevit_s,
    mobilevitwoF_s,
)
from model.backbone.conv_utils.ghost_conv import GhostModule, GhostBottleneck
from model.backbone.conv_utils.dcn import DeformableConv2d
from model.nets.Achelous import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prune_ratio = torch.rand(1) * 0.99


def test_l1():
    print("=" * 20, "test_BasePruner_with_example", "=" * 20)
    model = models.resnet18(pretrained=True)
    example_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    prune_ratio = 0.2
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    print((model(example_input) - out1).sum())


def test_resnet18_MMMTU():
    print("=" * 20, "test_resnet18_MMMTU", "=" * 20)
    model = models.resnet18(pretrained=False)
    example_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(
        model=model,
        example_input=example_input,
        algorithm="MMMTURatio",
        igtype2nodetype={},
        algo_args={
            "score_fn": "weight_sum_l1_out",
            "MMMTU": {
                "input_0": 1,
                "output_0": 1,
            },
        },
    )
    BP.prune(0.5)
    cal_flops(model, example_input, device)
    assert len(model(example_input)) == len(out1)


def test_BasePruner_with_Achelous_l1():
    print("=" * 20, "test_BasePruner_with_Achelous_MMMTU", "=" * 20)

    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )
    example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True),
        torch.randn(1, 3, 320, 320, requires_grad=True),
    ]
    cal_flops(model, example_input, device)
    out1 = model(*example_input)
    igtype2nodetype = {DeformableConv2d: dcnNode}
    BP = BasePruner(
        model=model,
        example_input=example_input,
        algorithm="UniformRatio",
        igtype2nodetype=igtype2nodetype,
        algo_args={
            "score_fn": "weight_sum_l1_out",
        },
    )
    BP.prune(0.5)
    cal_flops(model, example_input, device)
    assert len(model(*example_input)) == len(out1)


def test_BasePruner_with_Achelous_MMMTU():
    print("=" * 20, "test_BasePruner_with_Achelous_MMMTU", "=" * 20)

    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )
    example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True),
        torch.randn(1, 3, 320, 320, requires_grad=True),
    ]
    cal_flops(model, example_input, device)
    out1 = model(*example_input)
    igtype2nodetype = {DeformableConv2d: dcnNode}
    BP = BasePruner(
        model=model,
        example_input=example_input,
        # algorithm="UniformRatio",
        algorithm="MMMTURatio",
        igtype2nodetype=igtype2nodetype,
        algo_args={
            "score_fn": "weight_sum_l1_out",
            "MMMTU": {
                "input_0": 1,
                "input_1": 1,
                "output_0": 1,  # det_0
                "output_1": 1,  # det_1
                "output_2": 1,  # det_2
                "output_3": 1,  # se_seg
                "output_4": 1,  # lane_seg
            },
        },
    )
    BP.prune(0.5)
    cal_flops(model, example_input, device)
    assert len(model(*example_input)) == len(out1)


def test_BasePruner_with_Achelous_AMMU():
    print("=" * 20, "test_BasePruner_with_Achelous_AMMU", "=" * 20)
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )
    example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True),
        torch.randn(1, 3, 320, 320, requires_grad=True),
    ]
    cal_flops(model, example_input, device)
    out1 = model(*example_input)
    igtype2nodetype = {DeformableConv2d: dcnNode}
    BP = BasePruner(
        model=model,
        example_input=example_input,
        algorithm="AdaptiveMMU",
        igtype2nodetype=igtype2nodetype,
        algo_args={
            "score_fn": "weight_sum_l1_out",
            "model": model,
            "example_input": example_input,
        },
    )
    BP.prune(0.49)
    cal_flops(model, example_input, device)
    assert len(model(*example_input)) == len(out1)


if __name__ == "__main__":
    # test_l1()
    # test_resnet18_MMMTU()
    # test_BasePruner_with_Achelous_l1()
    # test_BasePruner_with_Achelous_MMMTU()
    test_BasePruner_with_Achelous_AMMU()
