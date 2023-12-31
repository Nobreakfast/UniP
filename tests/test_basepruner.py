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


def test_BasePruner_with_example():
    print("=" * 20, "test_BasePruner_with_example", "=" * 20)
    model = ExampleModel()
    example_input = torch.randn(1, 3, 4, 4, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert model(example_input).shape == out1.shape


def test_BasePruner_with_radarnet():
    print("=" * 20, "test_BasePruner_with_radarnet", "=" * 20)
    model = RCNet(in_channels=3)
    example_input = torch.randn(1, 3, 320, 320, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    igtype2nodetype = {DeformableConv2d: dcnNode}
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
        igtype2nodetype=igtype2nodetype,
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert len(model(example_input)) == len(out1)


def test_BasePruner_with_mvit():
    print("=" * 20, "test_BasePruner_with_mvit", "=" * 20)
    model = mobilevit_s(resolution=320)
    example_input = torch.randn(1, 3, 320, 320, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert model(example_input)[0].shape == out1[0].shape


def test_BasePruner_with_ghostbottleneck():
    print("=" * 20, "test_BasePruner_with_ghostmodule", "=" * 20)
    model = GhostBottleneck(in_chs=16, mid_chs=32, out_chs=32)
    example_input = torch.randn(1, 16, 256, 256, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert model(example_input).shape == out1.shape


def test_BasePruner_with_Achelous():
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
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
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert len(model(*example_input)) == len(out1)


def test_BasePruner_with_Achelous_only_radar():
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
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
    example_input = example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True),
        torch.randn(1, 3, 320, 320, requires_grad=True),
    ]
    cal_flops(model, example_input, device)
    out1 = model(*example_input)
    igtype2nodetype = {DeformableConv2d: dcnNode}
    BP = BasePruner(
        model=model.image_radar_encoder.radar_encoder,
        example_input=torch.randn(1, 3, 320, 320, requires_grad=True),
        algorithm="RandomRatio",
        igtype2nodetype=igtype2nodetype,
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert len(model(*example_input)) == len(out1)


def test_BasePruner_with_resnet18():
    print("=" * 20, "test_BasePruner_with_resnet18", "=" * 20)
    model = models.resnet18()
    example_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    cal_flops(model, example_input, device)
    out1 = model(example_input)
    BP = BasePruner(
        model,
        example_input,
        "UniformRatio",
        algo_args={"score_fn": "weight_sum_l1_out"},
    )
    BP.prune(prune_ratio)
    cal_flops(model, example_input, device)
    assert len(model(example_input)) == len(out1)


if __name__ == "__main__":
    test_BasePruner_with_example()
    # test_BasePruner_with_radarnet()
    # test_BasePruner_with_mvit()
    # test_BasePruner_with_ghostbottleneck()
    # test_BasePruner_with_Achelous()
    # test_BasePruner_with_Achelous_only_radar()
    # test_BasePruner_with_resnet18()
    # test_l1()
    # test_BasePruner_with_Achelous_MTU()
