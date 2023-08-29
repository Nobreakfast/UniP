import platform

if platform.system() != "Linux":
    exit(0)
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import unip
from unip.utils.energy import Calculator
from unip.utils.evaluation import cal_flops
from model.nets.Achelous import *


calculator = Calculator(cpu=True, device_id=0)


phi_list = ["S0", "S1", "S2"]
backbone_list = ["mv", "ef", "en", "ev", "rv", "pf"]
neck_list = ["gdf", "cdf"]


@calculator.measure(times=2000, warmup=1000)
def inference(model, example_input):
    model(*example_input)

device = torch.device("cuda:0")

def Achelous_energy(phi, backbone, neck):
    example_input = [
        torch.randn(1, 3, 320, 320),
        torch.randn(1, 3, 320, 320)
    ]
    model = Achelous3T(
            resolution=320,
            num_det=7,
            num_seg=9,
            phi=phi,
            backbone=backbone,
            neck=neck,
            spp=True,
            nano_head=False,
            )
    flops, params = cal_flops(model, example_input, 'cpu')
    example_input[0] = example_input[0].to(device)
    example_input[1] = example_input[1].to(device)
    model.to(device)
    model.eval()

    inference(model, example_input)
    p, e, pg, eg, pc, ec = calculator.summary(verbose=False)
    return p, e, pg, eg, pc, ec, flops, params


if __name__ == "__main__":
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    results = []
    for phi in phi_list:
        for backbone in backbone_list:
            for neck in neck_list:
                (
                    power,
                    energy,
                    gpu_power,
                    gpu_energy,
                    cpu_power,
                    cpu_energy,
                    flops,
                    params,
                ) = Achelous_energy(phi, backbone, neck)
                results.append([backbone+'-'+neck+'-'+phi, power, gpu_power, cpu_power, energy, gpu_energy, cpu_energy, flops, params])
    # save results to csv
    results = np.array(results)
    np.savetxt(
        "/home/allen/Downloads/Achelous_energy.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="backbone-neck-phi,power(mW),gpu_power(mW),cpu_power(mW),energy(J),gpu_energy(J),cpu_energy(J),flops,params",
    )
