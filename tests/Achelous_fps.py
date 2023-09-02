import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import unip
from unip.utils.evaluation import cal_flops, cal_fps
from model.nets.Achelous import *
import time
from tqdm import trange


phi_list = ["S0", "S1", "S2"]
backbone_list = ["mv", "ef", "en", "ev", "rv", "pf"]
neck_list = ["gdf", "cdf"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    # t, fps = inference(model, example_input)
    fps = cal_fps(model, example_input, device, times=1000, warmup=400)
    print(phi, backbone, neck, "time:", 1 / fps, "fps", fps)
    return [f"{phi}-{backbone}-{neck}", 1 / fps, fps, flops, params]


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
