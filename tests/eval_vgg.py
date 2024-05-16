import torch
from torchvision.models import vgg16

from unip.core.pruner import OneShotPruner
from unip.utils.evaluation import cal_flops


def eval_vgg16():
    model = vgg16()
    example_input = torch.rand(1, 3, 224, 224)
    cal_flops(model, example_input, device="cpu")
    pruner = OneShotPruner(model, example_input, ratio=0.5)
    pruner.plot()
    pruner.prune()
    cal_flops(model, example_input, device="cpu")
    output = model(example_input)
    print(output.shape)


if __name__ == "__main__":
    eval_vgg16()
