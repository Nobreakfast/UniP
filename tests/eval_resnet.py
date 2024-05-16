import torch
from torchvision.models import resnet18

# from unip.core.pruner import OneShotPruner, PPaIPruner
import unip
from unip.utils.evaluation import cal_flops


def eva_resnet18():
    model = resnet18()
    example_input = torch.rand(1, 3, 224, 224)
    cal_flops(model, example_input, device="cpu")
    # pruner = OneShotPruner(model, example_input, ratio=0.5)
    # pruner = PPaIPruner(model, example_input, pai="synflow", ratio=0.8)
    pruner = unip.prune("OneShot", model, example_input, ratio=0.8)
    # pruner.plot()
    pruner.prune()
    cal_flops(model, example_input, device="cpu")
    output = model(example_input)
    print(output.shape)


if __name__ == "__main__":
    eva_resnet18()
