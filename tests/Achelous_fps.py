import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import unip
from unip.utils.evaluation import cal_flops
from model.nets.Achelous import *
import time
from tqdm import trange


phi_list = ["S0", "S1", "S2"]
backbone_list = ["mv", "ef", "en", "ev", "rv", "pf"]
neck_list = ["gdf", "cdf"]

device = torch.device("cuda:0")


def inference(model, example_input, times=1000, warmup=400):
    for i in trange(warmup):
        model(*example_input)
    start = time.time()
    for i in trange(times):
        model(*example_input)
    end = time.time()
    return (end - start) / times, times / (end - start)


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
    flops, params = cal_flops(model, example_input, "cpu")
    example_input[0] = example_input[0].to(device)
    example_input[1] = example_input[1].to(device)
    model.to(device)
    model.eval()

    t, fps = inference(model, example_input)
    print(phi, backbone, neck, "time:", t, "fps", fps)
    return [f"{phi}-{backbone}-{neck}", t, fps, flops, params]


if __name__ == "__main__":
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    results = []
    for phi in phi_list:
        for backbone in backbone_list:
            for neck in neck_list:
                results.append(Achelous_energy(phi, backbone, neck))
    # save results to csv
    results = np.array(results)
    np.savetxt(
        "Achelous_fps.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="model_name,time,fps,flops,params",
    )
