import torch
import torch.nn as nn

CONV_TYPE = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)
NORM_TYPE = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
)
LINEAR_TYPE = (nn.Linear,)

POOLING_BACKWARD_TYPE = [
    "MaxPool2DWithIndicesBackward0",
    "AvgPool2DBackward0",
]
ACTIVITION_BACKWARD_TYPE = [
    "ReluBackward0",
    "SiluBackward0",
    "GeluBackward0",
    "HardswishBackward0",
    "SigmoidBackward0",
    "TanhBackward0",
    "SoftmaxBackward0",
    "LogSoftmaxBackward0",
    # not activation, but plays the same role
]
RESHAP_BACKWARD_TYPE = [
    "ReshapeAliasBackward0",
    "ViewBackward0",
    "UnsafeViewBackward0",
]

IGNORE_BACKWARD_TYPE = (
    "TBackward0",
    "NoneType",
)

PASS_BACKWARD_TYPE = ("CloneBackward0",)
MM_BACKWARD_TYPE = (
    "MmBackward0",
    "BmmBackward0",
)
UPSAMPLE_BACKWARD_TYPE = (
    "UpsampleBilinear2DBackward0",
    "UpsampleNearest2DBackward0",
    "UpsampleBicubic2DBackward0",
)

IDX_IN = 0
IDX_OUT = 1
DIM_IN = 1
DIM_OUT = 0
