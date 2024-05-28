from re import VERBOSE
import torch
from torchvision.models import resnet18, resnet50

# from unip.core.pruner import OneShotPruner, PPaIPruner
import unip
from unip.utils.evaluation import cal_flops

VERBOSE = False


def eva_resnet18():
    model = resnet18()
    example_input = torch.rand(1, 3, 224, 224, requires_grad=True)
    cal_flops(model, example_input, device="cpu")
    ignore_modules = {
        model.conv1: None,
        model.layer1[0].conv1: None,
    }
    # pruner = OneShotPruner(model, example_input, ratio=0.5)
    # pruner = PPaIPruner(model, example_input, pai="synflow", ratio=0.8)
    pruner = unip.prune(
        "OneShot",
        model,
        example_input,
        ratio=0.8,
        verbose=VERBOSE,
        ignore_modules=ignore_modules,
    )
    # pruner = unip.prune(
    # "OneShot", model, example_input, algorithm="gn", ratio=0.8, verbose=VERBOSE
    # )
    # pruner.plot(group=False)
    pruner.prune()
    cal_flops(model, example_input, device="cpu")
    output = model(example_input)
    print(output.shape)
    # print(model)


if __name__ == "__main__":
    eva_resnet18()
