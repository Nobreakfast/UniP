import platform

if platform.system() != "Linux":
    exit(0)
import torch
import torch.nn as nn
import torchvision.models as models
import unip
from unip.utils.energy import Calculator
from model.nets.Achelous import *


calculator = Calculator(cpu=True, device_id=0)


phi_list = ["S0"]  # , "S1", "S2"]
backbone_list = ["mv"]  # , "ef", "en", "ev", "rv", "pf"]
neck_list = ["gdf"]  # , "cdf"]


def test_Achelous_energy():
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True).cuda(),
        torch.randn(1, 3, 320, 320, requires_grad=True).cuda(),
    ]

    @calculator.measure(times=2000, warmup=500)
    def inference(model, example_input):
        model(*example_input)

    results = []
    for phi in phi_list:
        for backbone in backbone_list:
            for neck in neck_list:
                model = (
                    Achelous3T(
                        resolution=320,
                        num_det=7,
                        num_seg=9,
                        phi=phi,
                        backbone=backbone,
                        neck=neck,
                        spp=True,
                        nano_head=False,
                    )
                    .eval()
                    .cuda()
                )
                inference(model, example_input)
                power, energy = calculator.summary(verbose=False)
                results.append([phi, backbone, neck, power, energy])


if __name__ == "__main__":
    test_Achelous_energy()
