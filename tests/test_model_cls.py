import torch
import torch.nn as nn
import torchvision.models as models

import unip
from unip.core.pruner import BasePruner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prune_ratio = torch.rand(1)


def prune(model):
    example_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    BP = BasePruner(
        model,
        example_input,
        "RandomRatio",
        algo_args={"score_fn": "randn"},
    )
    BP.algorithm.run(prune_ratio)
    BP.prune()
    model(example_input)


# BUG: Linear
# def test_alexnet():
#     model = models.AlexNet()
#     prune(model)


def test_efficientnet():
    prune(models.efficientnet_b0())


def test_efficientnetv2():
    prune(models.efficientnet_v2_s())


def test_googlenet():
    prune(models.googlenet())


def test_resnet():
    prune(models.resnet18())


def test_wideresnet():
    prune(models.wide_resnet50_2())


# BUG: and input channel larger than groups
# def test_resnext():
#     prune(models.resnext101_32x8d())


def test_vgg():
    prune(models.vgg11())


# BUG: maybe reshape or permute
# def test_shufflenetv2():
#     prune(models.shufflenet_v2_x0_5())


def test_mobilenetv2():
    prune(models.mobilenet_v2())


def test_mobilenetv3():
    prune(models.mobilenet_v3_small())


# BUG:
# def test_inception():
#     prune(models.inception_v3())


# BUG: input channel larger than groups
# def test_regnet():
#     prune(models.regnet_y_400mf())


def test_squeezenet():
    prune(models.squeezenet1_0())


# BUG: linear keyerror: input
# def test_swintransformer():
#     prune(models.swin_t())


# BUG: linear keyerror: input
# def test_vit():
#     prune(models.vit_b_16())


if __name__ == "__main__":
    # test_alexnet()
    # test_efficientnet()
    # test_efficientnetv2()
    test_googlenet()
    # test_resnet()
    # test_wideresnet()
    # test_resnext()
    # test_vgg()
    # test_shufflenetv2()
    # test_mobilenetv2()
    # test_mobilenetv3()
    # test_inception()
    # test_regnet()
    # test_squeezenet()
    # test_swintransformer()
    # test_vit()
