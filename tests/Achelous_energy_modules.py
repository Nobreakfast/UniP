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


def forward_hook_record_input(module, input, output):
    setattr(module, "input", input[0])


def Achelous_energy(phi, backbone, neck):
    example_input = [torch.randn(1, 3, 320, 320), torch.randn(1, 3, 320, 320)]
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
    hooks = []
    for m in model.modules:
        hooks.append(m.register_forward_hook(forward_hook_record_input))
    model(*example_input)
    for h in hooks:
        h.remove()

    results = []
    for m in model.modules:
        example_input = torch.randn_list(m.input)
        flops, params = cal_flops(m, example_input, "cpu")
        example_input = example_input.to(device)
        m.to(device)
        m.eval()
        inference(m, example_input)
        p, e, pg, eg, pc, ec = calculator.summary(verbose=False)
        results.append([m, p, e, pg, eg, pc, ec, flops, params])
    results = np.array(results)
    np.savetxt(
        f"/home/allen/Downloads/AE/{backbone}-{neck}-{phi}.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="module,power(mW),gpu_power(mW),cpu_power(mW),energy(J),gpu_energy(J),cpu_energy(J),flops,params",
    )
    # return p, e, pg, eg, pc, ec, flops, params


if __name__ == "__main__":
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    # results = []
    for phi in phi_list:
        for backbone in backbone_list:
            for neck in neck_list:
                Achelous_energy(phi, backbone, neck)
